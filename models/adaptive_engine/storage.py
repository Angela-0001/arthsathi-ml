"""
Persistent weight storage for the adaptive bandit engine.
Stores per-item (alpha, beta) counts for Thompson Sampling.
Uses a JSON file — swap to SQLite or Redis in production.
"""

import json
from pathlib import Path
from threading import Lock

DEFAULT_PATH = Path("models/adaptive_engine/weights.json")
_lock = Lock()


def load(path: Path = DEFAULT_PATH) -> dict:
    """Returns {item_id: {"alpha": float, "beta": float}}"""
    if not path.exists():
        return {}
    with _lock:
        return json.loads(path.read_text(encoding="utf-8"))


def save(weights: dict, path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        path.write_text(json.dumps(weights, indent=2, ensure_ascii=False), encoding="utf-8")


def get_or_init(item_id: str, weights: dict) -> dict:
    """Return existing entry or initialise with uniform prior (alpha=1, beta=1)."""
    if item_id not in weights:
        weights[item_id] = {"alpha": 1.0, "beta": 1.0}
    return weights[item_id]
