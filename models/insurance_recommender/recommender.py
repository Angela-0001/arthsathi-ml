"""
Insurance Recommendation Model.

Pipeline:
  1. Hard eligibility filter (rule-based, deterministic)
  2. Semantic similarity ranking (sentence embedding cosine similarity)
  3. Adaptive score multiplier from BanditWeightEngine
  4. Returns ranked list with explanations

Member B owns this file.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from models.adaptive_engine.engine import BanditWeightEngine

ARTIFACTS_DIR = Path("models/insurance_recommender/artifacts")
WEIGHTS_PATH  = Path("models/adaptive_engine/insurance_weights.json")


class InsuranceRecommender:
    def __init__(self):
        self._records: List[dict] = []
        self._embeddings: np.ndarray | None = None
        self._embed_model = None
        self._bandit = BanditWeightEngine(weights_path=str(WEIGHTS_PATH))
        self._load_artifacts()

    def _load_artifacts(self):
        records_path = ARTIFACTS_DIR / "insurance_records.pkl"
        embeddings_path = ARTIFACTS_DIR / "insurance_embeddings.npy"

        if records_path.exists():
            with open(records_path, "rb") as f:
                self._records = pickle.load(f)

        if embeddings_path.exists():
            self._embeddings = np.load(embeddings_path)

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        return self._embed_model

    def recommend(self, user_profile: dict, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Returns top_k insurance schemes for a user profile.
        user_profile keys: age, gender, occupation, state, annual_income,
                           bank_account, aadhaar, goals (list of strings)
        """
        if not self._records:
            return []

        # Step 1: Hard eligibility filter
        eligible = [r for r in self._records if self._is_eligible(user_profile, r)]
        if not eligible:
            return []

        # Step 2: Semantic similarity to user's goals
        goal_text = " ".join(user_profile.get("goals", [])) or "insurance protection coverage"
        scores = self._semantic_scores(goal_text, eligible)

        # Step 3: Apply adaptive bandit weight
        results = []
        for i, record in enumerate(eligible):
            item_id = record.get("id", record.get("scheme_id", f"ins_{i}"))
            bandit_score = self._bandit.rank([item_id], n=1)[0][1] if item_id else 0.5
            final_score = scores[i] * (0.7 + 0.3 * bandit_score)  # blend semantic + bandit

            results.append({
                "id": item_id,
                "name": record.get("name", ""),
                "insurance_type": record.get("insurance_type", ""),
                "description": record.get("description", ""),
                "benefits": record.get("benefits", ""),
                "premium_annual": record.get("premium_annual"),
                "sum_assured": record.get("sum_assured"),
                "application_url": record.get("application_url", ""),
                "score": round(final_score, 4),
                "why": self._explain(user_profile, record),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def record_interaction(self, item_id: str, interaction_type: str):
        """Call this when user clicks / applies / dismisses a recommendation."""
        self._bandit.record(item_id, interaction_type)

    def _is_eligible(self, user: dict, scheme: dict) -> bool:
        age = user.get("age")
        income = user.get("annual_income")
        occupation = str(user.get("occupation", "")).lower()
        state = str(user.get("state", "")).lower()
        gender = str(user.get("gender", "all")).lower()
        has_bank = user.get("bank_account", True)
        has_aadhaar = user.get("aadhaar", True)

        if scheme.get("min_age") and age and age < scheme["min_age"]:
            return False
        if scheme.get("max_age") and age and age > scheme["max_age"]:
            return False
        if scheme.get("max_income_annual") and income and income > scheme["max_income_annual"]:
            return False
        eligible_occs = [o.lower() for o in (scheme.get("eligible_occupations") or [])]
        if eligible_occs and occupation and occupation not in eligible_occs:
            return False
        eligible_states = [s.lower() for s in (scheme.get("eligible_states") or [])]
        if eligible_states and state and state not in eligible_states:
            return False
        scheme_gender = str(scheme.get("gender", "all")).lower()
        if scheme_gender not in ("all", gender):
            return False
        if scheme.get("bank_account_required") and not has_bank:
            return False
        if scheme.get("aadhaar_required") and not has_aadhaar:
            return False
        return True

    def _semantic_scores(self, query: str, records: List[dict]) -> np.ndarray:
        """Cosine similarity between query embedding and record embeddings."""
        model = self._get_embed_model()
        query_emb = model.encode([query], normalize_embeddings=True)
        texts = [r.get("embed_text", r.get("name", "")) for r in records]
        record_embs = model.encode(texts, normalize_embeddings=True, batch_size=32)
        scores = (query_emb @ record_embs.T).flatten()
        return scores

    def _explain(self, user: dict, scheme: dict) -> str:
        """Generate a one-line human-readable reason for this recommendation."""
        reasons = []
        ins_type = scheme.get("insurance_type", "")
        premium = scheme.get("premium_annual")
        sum_assured = scheme.get("sum_assured")

        if premium == 0:
            reasons.append("free for you")
        elif premium and premium < 500:
            reasons.append(f"only ₹{premium:.0f}/year premium")
        if sum_assured:
            reasons.append(f"covers ₹{sum_assured:,.0f}")
        if ins_type:
            reasons.append(f"{ins_type} coverage")

        return "Recommended because: " + ("; ".join(reasons) if reasons else "matches your profile")
