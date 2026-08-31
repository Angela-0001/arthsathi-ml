"""
Evaluate insurance recommender: precision@k against a labeled test set.
Member B owns this file.

Test set format (data/insurance/test_profiles.jsonl):
  {"user_profile": {...}, "relevant_scheme_ids": ["PMJJBY", "PMSBY"]}
"""

import json
from pathlib import Path
from models.insurance_recommender.recommender import InsuranceRecommender

TEST_PATH = Path("data/insurance/test_profiles.jsonl")


def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    top_k = [r["id"] for r in recommended[:k]]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def run():
    if not TEST_PATH.exists():
        print(f"[WARN] No test file at {TEST_PATH}. Create it manually.")
        _create_sample_test()
        return

    recommender = InsuranceRecommender()
    test_cases = [json.loads(l) for l in TEST_PATH.read_text().splitlines() if l.strip()]

    p_at_1 = p_at_3 = p_at_5 = 0.0
    for case in test_cases:
        profile = case["user_profile"]
        relevant = case["relevant_scheme_ids"]
        results = recommender.recommend(profile, top_k=10)

        p_at_1 += precision_at_k(results, relevant, 1)
        p_at_3 += precision_at_k(results, relevant, 3)
        p_at_5 += precision_at_k(results, relevant, 5)

    n = len(test_cases)
    print(f"Evaluated on {n} test profiles")
    print(f"Precision@1 : {p_at_1/n:.4f}")
    print(f"Precision@3 : {p_at_3/n:.4f}")
    print(f"Precision@5 : {p_at_5/n:.4f}")


def _create_sample_test():
    """Create a small sample test file for quick smoke test."""
    samples = [
        {
            "user_profile": {
                "age": 30, "gender": "male", "occupation": "farmer",
                "state": "Maharashtra", "annual_income": 120000,
                "bank_account": True, "aadhaar": True,
                "goals": ["crop protection", "accident insurance"]
            },
            "relevant_scheme_ids": ["PMFBY", "PMSBY"]
        },
        {
            "user_profile": {
                "age": 25, "gender": "female", "occupation": "daily_wage_worker",
                "state": "Bihar", "annual_income": 60000,
                "bank_account": True, "aadhaar": True,
                "goals": ["health insurance", "life insurance"]
            },
            "relevant_scheme_ids": ["PMJAY", "PMJJBY"]
        },
    ]
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✓ Sample test file created: {TEST_PATH}")
    print("  Run this script again to evaluate.")


if __name__ == "__main__":
    run()
