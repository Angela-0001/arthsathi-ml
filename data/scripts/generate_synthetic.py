"""
Fallback: generate synthetic scheme + insurance records if real data collection fails.
Also used to AUGMENT real data with edge-case user profiles for model training.

Run this AFTER collect_schemes.py and collect_insurance.py to add synthetic samples.
Output:
  data/schemes/synthetic_schemes.jsonl
  data/insurance/synthetic_insurance.jsonl
  data/training/user_scheme_interactions.jsonl   ← for training the recommender
"""

import json
import random
from pathlib import Path

random.seed(42)

SCHEMES_OUT = Path("data/schemes/synthetic_schemes.jsonl")
INSURANCE_OUT = Path("data/insurance/synthetic_insurance.jsonl")
INTERACTIONS_OUT = Path("data/training/user_scheme_interactions.jsonl")

SCHEMES_OUT.parent.mkdir(parents=True, exist_ok=True)
INSURANCE_OUT.parent.mkdir(parents=True, exist_ok=True)
INTERACTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)

STATES = ["Maharashtra", "Uttar Pradesh", "Bihar", "Rajasthan", "Madhya Pradesh",
          "West Bengal", "Karnataka", "Tamil Nadu", "Gujarat", "Odisha"]
OCCUPATIONS = ["farmer", "daily_wage_worker", "small_trader", "domestic_worker",
               "construction_worker", "weaver", "fisherman", "salaried_employee", "self_employed"]
CATEGORIES = ["agriculture", "education", "health", "housing", "livelihood",
               "social_welfare", "women_empowerment", "financial_inclusion"]
MINISTRIES = [
    "Ministry of Agriculture", "Ministry of Labour", "Ministry of Health",
    "Ministry of Housing", "Ministry of Women and Child Development",
    "Ministry of Finance", "Ministry of Rural Development",
]


def make_scheme(i: int) -> dict:
    cat = random.choice(CATEGORIES)
    occ = random.choices(OCCUPATIONS, k=random.randint(0, 3))
    has_income_limit = random.random() > 0.4
    return {
        "scheme_id": f"SYN_SCHEME_{i:04d}",
        "name": f"Synthetic {cat.replace('_', ' ').title()} Support Scheme {i}",
        "description": f"Government support scheme for {cat} beneficiaries, providing financial assistance.",
        "ministry": random.choice(MINISTRIES),
        "category": cat,
        "eligibility": [f"Must be {o}" for o in occ],
        "benefits": f"₹{random.choice([5000, 10000, 25000, 50000, 100000]):,} financial assistance",
        "min_age": random.choice([18, 21, None, None]),
        "max_age": random.choice([45, 60, None, None, None]),
        "max_income_annual": random.randint(100000, 300000) if has_income_limit else None,
        "eligible_states": random.choices(STATES, k=random.randint(0, 5)),
        "eligible_occupations": occ,
        "gender": random.choice(["all", "all", "all", "female", "male"]),
        "application_url": f"https://example.gov.in/scheme-{i}",
        "source": "synthetic",
    }


def make_user_profile() -> dict:
    age = random.randint(18, 70)
    income = random.randint(5000, 40000) * 12  # annual
    return {
        "user_id": f"USER_{random.randint(10000, 99999)}",
        "age": age,
        "gender": random.choice(["male", "female"]),
        "occupation": random.choice(OCCUPATIONS),
        "state": random.choice(STATES),
        "annual_income": income,
        "has_bank_account": random.random() > 0.2,
        "has_aadhaar": random.random() > 0.1,
    }


def generate_interactions(schemes: list[dict], n_users: int = 500) -> list[dict]:
    """
    Simulate user-scheme interactions for training the adaptive recommender.
    interaction_type: "viewed" | "applied" | "dismissed"
    """
    interactions = []
    for _ in range(n_users):
        user = make_user_profile()
        n_interactions = random.randint(3, 10)
        seen = random.choices(schemes, k=n_interactions)
        for s in seen:
            interaction_type = random.choices(
                ["viewed", "applied", "dismissed"],
                weights=[0.5, 0.3, 0.2]
            )[0]
            interactions.append({
                "user_id": user["user_id"],
                "scheme_id": s["scheme_id"],
                "interaction_type": interaction_type,
                "user_profile": user,
            })
    return interactions


def run():
    # Synthetic schemes
    synth_schemes = [make_scheme(i) for i in range(1, 201)]
    with open(SCHEMES_OUT, "w", encoding="utf-8") as f:
        for s in synth_schemes:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✓ {len(synth_schemes)} synthetic schemes → {SCHEMES_OUT}")

    # Synthetic insurance records
    synth_insurance = []
    insurance_types = ["health", "life", "accident", "crop"]
    for i in range(1, 51):
        t = random.choice(insurance_types)
        synth_insurance.append({
            "scheme_id": f"SYN_INS_{i:03d}",
            "name": f"Synthetic {t.title()} Insurance Scheme {i}",
            "type": t,
            "description": f"Provides {t} coverage for eligible citizens.",
            "premium_annual": random.choice([0, 20, 436, 1000, 2000]),
            "sum_assured": random.choice([30000, 100000, 200000, 500000]),
            "min_age": random.choice([18, 21]),
            "max_age": random.choice([55, 60, 70, None]),
            "max_income_annual": random.choice([None, 150000, 300000]),
            "eligible_states": random.choices(STATES, k=random.randint(0, 3)),
            "eligible_occupations": random.choices(OCCUPATIONS, k=random.randint(0, 2)),
            "gender": random.choice(["all", "all", "female"]),
            "source": "synthetic",
        })
    with open(INSURANCE_OUT, "w", encoding="utf-8") as f:
        for s in synth_insurance:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✓ {len(synth_insurance)} synthetic insurance records → {INSURANCE_OUT}")

    # User-scheme interactions
    all_schemes = synth_schemes  # use synthetic schemes; replace with real when available
    interactions = generate_interactions(all_schemes)
    with open(INTERACTIONS_OUT, "w", encoding="utf-8") as f:
        for rec in interactions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✓ {len(interactions)} user-scheme interactions → {INTERACTIONS_OUT}")


if __name__ == "__main__":
    run()
