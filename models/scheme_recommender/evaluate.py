"""
Evaluate scheme recommender: precision@k.
Member A owns this file.

Test set: data/schemes/test_profiles.jsonl
  {"user_profile": {...}, "relevant_scheme_ids": ["PM-KISAN", "PMEGP"]}
"""

import json
from pathlib import Path
from models.scheme_recommender.recommender import SchemeRecommender

TEST_PATH = Path("data/schemes/test_profiles.jsonl")


def precision_at_k(recommended, relevant, k):
    top_k = [r["id"] for r in recommended[:k]]
    return sum(1 for i in top_k if i in relevant) / k


def run():
    if not TEST_PATH.exists():
        _create_sample_test()
        return

    rec = SchemeRecommender()
    cases = [json.loads(l) for l in TEST_PATH.read_text().splitlines() if l.strip()]

    p1 = p3 = p5 = 0.0
    for case in cases:
        results = rec.recommend(case["user_profile"], top_k=10)
        relevant = case["relevant_scheme_ids"]
        p1 += precision_at_k(results, relevant, 1)
        p3 += precision_at_k(results, relevant, 3)
        p5 += precision_at_k(results, relevant, 5)

    n = len(cases)
    print(f"Scheme Recommender — {n} test profiles")
    print(f"Precision@1 : {p1/n:.4f}")
    print(f"Precision@3 : {p3/n:.4f}")
    print(f"Precision@5 : {p5/n:.4f}")


def _create_sample_test():
    samples = [
        {
            "user_profile": {
                "age": 40, "gender": "male", "occupation": "farmer",
                "state": "Punjab", "annual_income": 90000,
                "goals": ["agriculture support", "crop loan"]
            },
            "relevant_scheme_ids": ["PMKISAN", "KCC"]
        },
        {
            "user_profile": {
                "age": 22, "gender": "female", "occupation": "self_employed",
                "state": "Uttar Pradesh", "annual_income": 80000,
                "goals": ["business loan", "women empowerment"]
            },
            "relevant_scheme_ids": ["MUDRA", "PMEGP"]
        },
    ]
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✓ Sample test created: {TEST_PATH}. Run again to evaluate.")


if __name__ == "__main__":
    run()
