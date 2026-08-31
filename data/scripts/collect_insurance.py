"""
MEMBER B — Step 1: Collect insurance schemes data.

Sources:
  1. Hand-curated: all major GoI insurance schemes (PMJJBY, PMSBY, PMFBY, PM-JAY, etc.)
     — these are well-documented, finite, and we build the structured records ourselves
  2. IRDAI public data (irdai.gov.in) — scrape product listings
  3. HuggingFace: sujra/insurance_llama2 — LIC/general insurance Q&A corpus

Because no single clean structured insurance dataset exists publicly at the right
granularity, we build the GoI insurance records ourselves and supplement with IRDAI.

Output: data/insurance/raw_insurance.jsonl
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "insurance"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "raw_insurance.jsonl"

# ── Hand-curated GoI insurance schemes ──────────────────────────────────────
# These are fact-checked against official PIB / scheme portals.
# Eligibility fields are the actual criteria from scheme guidelines.
GOI_INSURANCE_SCHEMES = [
    {
        "scheme_id": "PMJJBY",
        "name": "Pradhan Mantri Jeevan Jyoti Bima Yojana",
        "type": "life",
        "description": "Renewable 1-year life insurance cover for death due to any reason. Premium ₹436/year.",
        "insurer": "LIC and other life insurers via banks",
        "premium_annual": 436,
        "sum_assured": 200000,
        "min_age": 18,
        "max_age": 50,
        "max_income_annual": None,  # no income limit
        "eligible_states": [],      # all states
        "eligible_occupations": [], # all
        "gender": "all",
        "bank_account_required": True,
        "aadhaar_required": False,
        "benefits": "₹2 lakh on death from any cause",
        "application_url": "https://www.jansuraksha.gov.in/",
        "tags": ["life_insurance", "death_benefit", "low_premium"],
        "source": "manual/pib",
    },
    {
        "scheme_id": "PMSBY",
        "name": "Pradhan Mantri Suraksha Bima Yojana",
        "type": "accident",
        "description": "1-year accidental death and disability insurance. Premium ₹20/year.",
        "insurer": "Public sector general insurers via banks",
        "premium_annual": 20,
        "sum_assured": 200000,
        "min_age": 18,
        "max_age": 70,
        "max_income_annual": None,
        "eligible_states": [],
        "eligible_occupations": [],
        "gender": "all",
        "bank_account_required": True,
        "aadhaar_required": False,
        "benefits": "₹2 lakh on accidental death or permanent disability; ₹1 lakh on partial disability",
        "application_url": "https://www.jansuraksha.gov.in/",
        "tags": ["accident_insurance", "disability", "low_premium"],
        "source": "manual/pib",
    },
    {
        "scheme_id": "PMFBY",
        "name": "Pradhan Mantri Fasal Bima Yojana",
        "type": "crop",
        "description": "Crop insurance for farmers. Premium: 2% for Kharif, 1.5% for Rabi, 5% for commercial crops.",
        "insurer": "Empanelled insurance companies",
        "premium_annual": None,  # percentage-based
        "sum_assured": None,     # varies by crop/state
        "min_age": 18,
        "max_age": None,
        "max_income_annual": None,
        "eligible_states": [],
        "eligible_occupations": ["farmer"],
        "gender": "all",
        "bank_account_required": True,
        "aadhaar_required": True,
        "benefits": "Compensation for crop loss due to natural calamities, pests, diseases",
        "application_url": "https://pmfby.gov.in/",
        "tags": ["crop_insurance", "farmer", "agriculture"],
        "source": "manual/pib",
    },
    {
        "scheme_id": "PMJAY",
        "name": "Pradhan Mantri Jan Arogya Yojana (Ayushman Bharat)",
        "type": "health",
        "description": "Health cover of ₹5 lakh per family per year for secondary and tertiary hospitalisation.",
        "insurer": "Government-funded (NHPM)",
        "premium_annual": 0,  # free for beneficiaries
        "sum_assured": 500000,
        "min_age": 0,
        "max_age": None,
        "max_income_annual": None,  # based on SECC/PMJAY list, not income
        "eligible_states": [],
        "eligible_occupations": [],
        "gender": "all",
        "bank_account_required": False,
        "aadhaar_required": True,
        "benefits": "₹5 lakh/family/year for hospitalization; cashless treatment at empanelled hospitals",
        "application_url": "https://pmjay.gov.in/",
        "tags": ["health_insurance", "hospitalization", "free", "family"],
        "source": "manual/pib",
    },
    {
        "scheme_id": "ESIC",
        "name": "Employees State Insurance Scheme",
        "type": "health",
        "description": "Medical, sickness, maternity, and disability benefits for low-wage employees.",
        "insurer": "ESIC (Government)",
        "premium_annual": None,  # percentage of wages
        "sum_assured": None,
        "min_age": 18,
        "max_age": None,
        "max_income_annual": 252000,  # ₹21,000/month limit
        "eligible_states": [],
        "eligible_occupations": ["salaried_employee"],
        "gender": "all",
        "bank_account_required": True,
        "aadhaar_required": True,
        "benefits": "Medical care, sickness cash, maternity benefit, disability pension",
        "application_url": "https://www.esic.in/",
        "tags": ["health_insurance", "salaried", "maternity", "disability"],
        "source": "manual/irdai",
    },
    {
        "scheme_id": "NAIS",
        "name": "National Agricultural Insurance Scheme",
        "type": "crop",
        "description": "Crop insurance for small and marginal farmers. Government-subsidised premium.",
        "insurer": "Agriculture Insurance Company of India (AIC)",
        "premium_annual": None,
        "sum_assured": None,
        "min_age": 18,
        "max_age": None,
        "max_income_annual": None,
        "eligible_states": [],
        "eligible_occupations": ["farmer", "small_farmer", "marginal_farmer"],
        "gender": "all",
        "bank_account_required": True,
        "aadhaar_required": False,
        "benefits": "Compensation for yield loss due to natural calamities",
        "application_url": "https://www.aicofindia.com/",
        "tags": ["crop_insurance", "farmer", "small_farmer"],
        "source": "manual/aic",
    },
    {
        "scheme_id": "RSBY",
        "name": "Rashtriya Swasthya Bima Yojana",
        "type": "health",
        "description": "Cashless health insurance for BPL families. Now largely subsumed by PM-JAY.",
        "insurer": "Government-funded",
        "premium_annual": 0,
        "sum_assured": 30000,
        "min_age": 0,
        "max_age": None,
        "max_income_annual": None,
        "eligible_states": [],
        "eligible_occupations": [],
        "gender": "all",
        "bank_account_required": False,
        "aadhaar_required": False,
        "benefits": "₹30,000/family/year hospitalization cover",
        "application_url": "https://www.rsby.gov.in/",
        "tags": ["health_insurance", "bpl", "cashless"],
        "source": "manual/labour_ministry",
    },
    {
        "scheme_id": "PMVVVY",
        "name": "Pradhan Mantri Vaya Vandana Yojana",
        "type": "pension_insurance",
        "description": "Pension scheme for senior citizens providing guaranteed return of 7.4% p.a. for 10 years.",
        "insurer": "LIC of India",
        "premium_annual": None,
        "sum_assured": 1500000,  # max investment
        "min_age": 60,
        "max_age": None,
        "max_income_annual": None,
        "eligible_states": [],
        "eligible_occupations": [],
        "gender": "all",
        "bank_account_required": True,
        "aadhaar_required": True,
        "benefits": "Monthly pension + death benefit; guaranteed 7.4% return",
        "application_url": "https://licindia.in/",
        "tags": ["senior_citizen", "pension", "guaranteed_return"],
        "source": "manual/lic",
    },
]


def run():
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for scheme in GOI_INSURANCE_SCHEMES:
            f.write(json.dumps(scheme, ensure_ascii=False) + "\n")
    print(f"✓ Saved {len(GOI_INSURANCE_SCHEMES)} insurance schemes to {OUT_FILE}")
    print("\nNext step: run scrape_irdai.py to supplement with private insurer products")


if __name__ == "__main__":
    run()
