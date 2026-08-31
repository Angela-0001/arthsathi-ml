"""
Scheme Recommendation Model.
Mirrors InsuranceRecommender but for government schemes.
Member A owns this file.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from models.adaptive_engine.engine import BanditWeightEngine

ARTIFACTS_DIR = Path("models/scheme_recommender/artifacts")
WEIGHTS_PATH  = Path("models/adaptive_engine/scheme_weights.json")


class SchemeRecommender:
    def __init__(self):
        self._records: List[dict] = []
        self._embed_model = None
        self._bandit = BanditWeightEngine(weights_path=str(WEIGHTS_PATH))
        self._load_artifacts()

    def _load_artifacts(self):
        p = ARTIFACTS_DIR / "scheme_records.pkl"
        if p.exists():
            with open(p, "rb") as f:
                self._records = pickle.load(f)

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        return self._embed_model

    def recommend(self, user_profile: dict, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self._records:
            return []

        eligible = [r for r in self._records if self._is_eligible(user_profile, r)]
        if not eligible:
            return []

        goal_text = " ".join(user_profile.get("goals", [])) or "government welfare scheme financial help"
        scores = self._semantic_scores(goal_text, eligible)

        results = []
        for i, record in enumerate(eligible):
            item_id = record.get("id", f"scheme_{i}")
            bandit_score = self._bandit.rank([item_id], n=1)[0][1]
            final_score = scores[i] * (0.7 + 0.3 * bandit_score)
            results.append({
                "id": item_id,
                "name": record.get("name", ""),
                "category": record.get("category", ""),
                "description": record.get("description", ""),
                "benefits": record.get("benefits", ""),
                "application_url": record.get("application_url", ""),
                "score": round(final_score, 4),
                "why": self._explain(user_profile, record),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def record_interaction(self, item_id: str, interaction_type: str):
        self._bandit.record(item_id, interaction_type)

    def _is_eligible(self, user: dict, scheme: dict) -> bool:
        age = user.get("age")
        income = user.get("annual_income")
        occupation = str(user.get("occupation", "")).lower()
        state = str(user.get("state", "")).lower()
        gender = str(user.get("gender", "all")).lower()

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
        return True

    def _semantic_scores(self, query: str, records: List[dict]) -> np.ndarray:
        model = self._get_embed_model()
        query_emb = model.encode([query], normalize_embeddings=True)
        texts = [r.get("embed_text", r.get("name", "")) for r in records]
        record_embs = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return (query_emb @ record_embs.T).flatten()

    def _explain(self, user: dict, scheme: dict) -> str:
        parts = []
        if scheme.get("category"):
            parts.append(f"{scheme['category']} scheme")
        if scheme.get("benefits"):
            parts.append(scheme["benefits"][:60])
        return "Recommended: " + "; ".join(parts) if parts else "Matches your profile"
