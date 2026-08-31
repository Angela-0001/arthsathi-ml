"""
Adaptive Recommendation Engine — Thompson Sampling Bandit.

Why Thompson Sampling:
  - Each scheme/insurance item has an unknown "true" click/apply rate
  - We maintain a Beta distribution Beta(alpha, beta) per item
  - alpha = successes (applied/clicked), beta = failures (dismissed/ignored)
  - At recommendation time, we SAMPLE from each item's distribution and rank
  - Items with high uncertainty get explored; proven items get exploited
  - Over time, weights converge to reflect what actually helps users

This directly addresses the research gap: existing systems use static ranking.
Ours learns continuously from real user choices.

Member B owns this file.
"""

import math
import random
from typing import List, Tuple
from models.adaptive_engine.storage import load, save, get_or_init


class BanditWeightEngine:
    """
    Thompson Sampling bandit for adaptive scheme/insurance ranking.

    Usage:
        engine = BanditWeightEngine()
        ranked = engine.rank(candidate_item_ids)
        engine.record(item_id, interaction_type)  # "applied" | "viewed" | "dismissed"
    """

    # Reward values for different interaction types
    REWARDS = {
        "applied":   1.0,   # user actually applied → strong positive signal
        "clicked":   0.3,   # user clicked for details → mild positive
        "viewed":    0.1,   # system showed it, user didn't dismiss → weak positive
        "dismissed": 0.0,   # user dismissed → negative (beta increment)
    }

    # Decay factor — old interactions matter less over time
    # Applied weekly: multiply alpha/beta by DECAY each week
    DECAY = 0.98

    def __init__(self, weights_path: str | None = None):
        from pathlib import Path
        self._path = Path(weights_path) if weights_path else None
        self._weights = load(self._path) if self._path else {}

    def rank(self, item_ids: List[str], n: int | None = None) -> List[Tuple[str, float]]:
        """
        Rank items by Thompson Sampling score.
        Returns list of (item_id, sampled_score) sorted descending.
        """
        scored = []
        for item_id in item_ids:
            entry = get_or_init(item_id, self._weights)
            alpha = entry["alpha"]
            beta  = entry["beta"]
            # Sample from Beta distribution
            score = _beta_sample(alpha, beta)
            scored.append((item_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n] if n else scored

    def record(self, item_id: str, interaction_type: str) -> None:
        """
        Update Beta distribution based on user interaction.
        applied/clicked → alpha += reward
        dismissed → beta += 1
        """
        entry = get_or_init(item_id, self._weights)
        reward = self.REWARDS.get(interaction_type, 0.0)

        if reward > 0:
            entry["alpha"] += reward
        else:
            entry["beta"] += 1.0

        # Cap to prevent runaway values
        entry["alpha"] = min(entry["alpha"], 1000.0)
        entry["beta"]  = min(entry["beta"],  1000.0)

        if self._path:
            save(self._weights, self._path)

    def decay_weights(self) -> None:
        """
        Apply temporal decay — call weekly via scheduler.
        Prevents stale interactions from dominating forever.
        """
        for item_id, entry in self._weights.items():
            entry["alpha"] = max(1.0, entry["alpha"] * self.DECAY)
            entry["beta"]  = max(1.0, entry["beta"]  * self.DECAY)
        if self._path:
            save(self._weights, self._path)

    def confidence_interval(self, item_id: str, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Return (lower, upper) bounds of the Beta credible interval.
        Useful for showing uncertainty to the user or in evaluation.
        """
        from scipy.stats import beta as beta_dist
        entry = get_or_init(item_id, self._weights)
        a, b = entry["alpha"], entry["beta"]
        lo = beta_dist.ppf((1 - confidence) / 2, a, b)
        hi = beta_dist.ppf(1 - (1 - confidence) / 2, a, b)
        return round(lo, 4), round(hi, 4)

    def get_stats(self, item_id: str) -> dict:
        entry = get_or_init(item_id, self._weights)
        a, b = entry["alpha"], entry["beta"]
        mean = a / (a + b)
        # Mode (most likely CTR estimate)
        mode = (a - 1) / (a + b - 2) if (a + b) > 2 else 0.5
        return {
            "item_id": item_id,
            "alpha": round(a, 3),
            "beta": round(b, 3),
            "estimated_ctr": round(mean, 4),
            "mode": round(mode, 4),
            "n_interactions": int(a + b - 2),  # subtract priors
        }


def _beta_sample(alpha: float, beta: float) -> float:
    """
    Sample from Beta(alpha, beta) using Python's random module.
    No scipy needed at runtime — uses the relation Beta = Gamma/Gamma.
    """
    # random.betavariate uses Knuth's method
    try:
        return random.betavariate(alpha, beta)
    except ValueError:
        return alpha / (alpha + beta)  # fallback to mean
