"""
Collect government schemes data.
Primary source: Aryan-Pardeshi/gov-myscheme-dataset (2066 schemes from myScheme portal)
Fallback: hand-curated 35+ schemes if HuggingFace unavailable.
"""

import json
from pathlib import Path

OUT_DIR = Path("data/schemes")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "raw_schemes.jsonl"

KNOWN_SCHEMES = [
    # ── Agriculture ──────────────────────────────────────────────────────────
    {
        "scheme_id": "PMKISAN",
        "name": "PM Kisan Samman Nidhi",
        "description": "Income support of Rs 6000/year to all farmer families owning cultivable land, paid in 3 installments of Rs 2000 each directly to bank account.",
        "ministry": "Ministry of Agriculture & Farmers Welfare",
        "category": "agriculture",
        "benefits": "Rs 6000 per year in 3 installments of Rs 2000 each",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": ["farmer"],
        "gender": "all", "application_url": "https://pmkisan.gov.in/",
        "tags": ["farmer", "income_support", "agriculture"],
    },
    {
        "scheme_id": "KISAN_CREDIT",
        "name": "Kisan Credit Card",
        "description": "Short-term credit for farmers to meet agricultural and allied needs at subsidised interest rate of 4% per annum.",
        "ministry": "Ministry of Agriculture & Farmers Welfare",
        "category": "agriculture",
        "benefits": "Credit up to Rs 3 lakh at 4% interest with 3% government subsidy",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": ["farmer"],
        "gender": "all", "application_url": "https://www.nabard.org/",
        "tags": ["farmer", "credit", "loan", "agriculture"],
    },
    {
        "scheme_id": "PMKSY",
        "name": "Pradhan Mantri Krishi Sinchayee Yojana",
        "description": "Irrigation support to ensure water to every farm field. Subsidises drip and sprinkler irrigation systems.",
        "ministry": "Ministry of Agriculture & Farmers Welfare",
        "category": "agriculture",
        "benefits": "Drip/sprinkler irrigation subsidy up to 55% for small and marginal farmers",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": ["farmer"],
        "gender": "all", "application_url": "https://pmksy.gov.in/",
        "tags": ["irrigation", "farmer", "water", "subsidy"],
    },
    {
        "scheme_id": "SOIL_HEALTH",
        "name": "Soil Health Card Scheme",
        "description": "Provides farmers with soil health cards containing crop-wise recommendations for nutrients and fertilisers.",
        "ministry": "Ministry of Agriculture & Farmers Welfare",
        "category": "agriculture",
        "benefits": "Free soil testing and personalised fertiliser recommendations",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": ["farmer"],
        "gender": "all", "application_url": "https://soilhealth.dac.gov.in/",
        "tags": ["farmer", "soil", "free_service"],
    },
    {
        "scheme_id": "PMFBY",
        "name": "Pradhan Mantri Fasal Bima Yojana",
        "description": "Crop insurance for farmers. Premium: 2% for Kharif crops, 1.5% for Rabi crops, 5% for commercial/horticulture crops.",
        "ministry": "Ministry of Agriculture & Farmers Welfare",
        "category": "agriculture",
        "benefits": "Compensation for crop loss due to natural calamities, pests, diseases",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": ["farmer"],
        "gender": "all", "application_url": "https://pmfby.gov.in/",
        "tags": ["crop_insurance", "farmer", "agriculture"],
    },
    {
        "scheme_id": "AGRI_INFRA",
        "name": "Agriculture Infrastructure Fund",
        "description": "Medium-long term debt financing for post-harvest management and community farm assets.",
        "ministry": "Ministry of Agriculture & Farmers Welfare",
        "category": "agriculture",
        "benefits": "Loans up to Rs 2 crore with 3% interest subvention and credit guarantee",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": ["farmer", "self_employed"],
        "gender": "all", "application_url": "https://agriinfra.dac.gov.in/",
        "tags": ["farmer", "infrastructure", "loan", "storage"],
    },

    # ── Housing ──────────────────────────────────────────────────────────────
    {
        "scheme_id": "PMAWAS_G",
        "name": "Pradhan Mantri Awaas Yojana - Gramin",
        "description": "Financial assistance for construction of pucca houses for rural homeless and those living in kutcha houses.",
        "ministry": "Ministry of Rural Development",
        "category": "housing",
        "benefits": "Rs 1.2 lakh in plain areas, Rs 1.3 lakh in hilly/NE/IAP areas",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://pmayg.nic.in/",
        "tags": ["housing", "rural", "construction", "bpl"],
    },
    {
        "scheme_id": "PMAWAS_U",
        "name": "Pradhan Mantri Awaas Yojana - Urban",
        "description": "Housing for all urban poor. Interest subsidy on home loans for EWS, LIG, and MIG categories.",
        "ministry": "Ministry of Housing and Urban Affairs",
        "category": "housing",
        "benefits": "Interest subsidy of 3-6.5% on home loans; EWS gets Rs 1.5 lakh subsidy",
        "min_age": 18, "max_age": None, "max_income_annual": 1800000,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://pmaymis.gov.in/",
        "tags": ["housing", "urban", "home_loan", "subsidy"],
    },

    # ── Livelihood & Employment ───────────────────────────────────────────────
    {
        "scheme_id": "NREGS",
        "name": "Mahatma Gandhi National Rural Employment Guarantee Scheme",
        "description": "Guarantees 100 days of wage employment per financial year to every rural household whose adult members volunteer to do unskilled manual work.",
        "ministry": "Ministry of Rural Development",
        "category": "livelihood",
        "benefits": "100 days guaranteed employment at state minimum wage",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": ["daily_wage_worker"],
        "gender": "all", "application_url": "https://nrega.nic.in/",
        "tags": ["employment", "rural", "wage", "guaranteed"],
    },
    {
        "scheme_id": "PMEGP",
        "name": "Prime Minister Employment Generation Programme",
        "description": "Credit-linked subsidy for setting up micro enterprises in non-farm sector for generating self-employment opportunities.",
        "ministry": "Ministry of MSME",
        "category": "livelihood",
        "benefits": "15-35% subsidy on project cost up to Rs 25 lakh (manufacturing) and Rs 10 lakh (services)",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": ["self_employed", "small_trader"],
        "gender": "all", "application_url": "https://www.kviconline.gov.in/",
        "tags": ["self_employment", "subsidy", "enterprise", "msme"],
    },
    {
        "scheme_id": "MUDRA",
        "name": "Pradhan Mantri MUDRA Yojana",
        "description": "Loans up to Rs 10 lakh for non-corporate, non-farm small and micro enterprises through Mudra loans.",
        "ministry": "Ministry of Finance",
        "category": "financial_inclusion",
        "benefits": "Shishu: up to Rs 50000 | Kishore: Rs 50000-5 lakh | Tarun: Rs 5-10 lakh",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": ["self_employed", "small_trader"],
        "gender": "all", "application_url": "https://www.mudra.org.in/",
        "tags": ["loan", "micro_enterprise", "self_employment"],
    },
    {
        "scheme_id": "PM_SVANidhi",
        "name": "PM Street Vendor's AtmaNirbhar Nidhi (SVANidhi)",
        "description": "Micro-credit scheme for street vendors to resume livelihoods. Collateral-free working capital loans.",
        "ministry": "Ministry of Housing and Urban Affairs",
        "category": "livelihood",
        "benefits": "Working capital loan of Rs 10000 extendable to Rs 20000 and Rs 50000",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": ["street_vendor", "small_trader"],
        "gender": "all", "application_url": "https://pmsvanidhi.mohua.gov.in/",
        "tags": ["street_vendor", "micro_credit", "urban"],
    },
    {
        "scheme_id": "DEEN_DAYAL_ANTYODAYA",
        "name": "Deen Dayal Antyodaya Yojana - National Rural Livelihoods Mission",
        "description": "Skill training and placement for rural poor. Organises rural poor into Self Help Groups.",
        "ministry": "Ministry of Rural Development",
        "category": "livelihood",
        "benefits": "Free skill training, Rs 15000-35000 certificate courses, placement support, SHG formation",
        "min_age": 15, "max_age": 35, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://aajeevika.gov.in/",
        "tags": ["skill_training", "employment", "youth", "shg"],
    },
    {
        "scheme_id": "SKILL_INDIA",
        "name": "Pradhan Mantri Kaushal Vikas Yojana",
        "description": "Skill certification scheme to enable Indian youth to take up industry-relevant skill training.",
        "ministry": "Ministry of Skill Development",
        "category": "livelihood",
        "benefits": "Free skill training with Rs 8000 average reward on certification",
        "min_age": 15, "max_age": 45, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://www.pmkvyofficial.org/",
        "tags": ["skill_training", "youth", "certification", "employment"],
    },
    {
        "scheme_id": "STARTUP_INDIA",
        "name": "Startup India Seed Fund Scheme",
        "description": "Financial assistance for startups for proof of concept, prototype development, product trials.",
        "ministry": "Ministry of Commerce and Industry",
        "category": "livelihood",
        "benefits": "Up to Rs 20 lakh for proof of concept; Rs 50 lakh for market entry",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": ["self_employed"],
        "gender": "all", "application_url": "https://www.startupindia.gov.in/",
        "tags": ["startup", "entrepreneur", "seed_fund"],
    },

    # ── Financial Inclusion ───────────────────────────────────────────────────
    {
        "scheme_id": "PMJDY",
        "name": "Pradhan Mantri Jan Dhan Yojana",
        "description": "Zero balance bank account with RuPay debit card, Rs 2 lakh accident cover, and Rs 30000 life cover.",
        "ministry": "Ministry of Finance",
        "category": "financial_inclusion",
        "benefits": "Zero balance account, Rs 2 lakh accident insurance, Rs 30000 life cover, overdraft up to Rs 10000",
        "min_age": 10, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://pmjdy.gov.in/",
        "tags": ["bank_account", "financial_inclusion", "insurance"],
    },
    {
        "scheme_id": "ATAL_PENSION",
        "name": "Atal Pension Yojana",
        "description": "Guaranteed pension of Rs 1000-5000/month after age 60 for unorganised sector workers.",
        "ministry": "Ministry of Finance",
        "category": "financial_inclusion",
        "benefits": "Rs 1000 to Rs 5000 guaranteed monthly pension after age 60",
        "min_age": 18, "max_age": 40, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://npscra.nsdl.co.in/",
        "tags": ["pension", "unorganised_sector", "retirement"],
    },
    {
        "scheme_id": "STAND_UP_INDIA",
        "name": "Stand-Up India",
        "description": "Bank loans between Rs 10 lakh and Rs 1 crore to SC/ST and women entrepreneurs for greenfield enterprises.",
        "ministry": "Ministry of Finance",
        "category": "financial_inclusion",
        "benefits": "Loan Rs 10 lakh to Rs 1 crore at composite loan for greenfield project",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": ["self_employed"],
        "gender": "all", "application_url": "https://www.standupmitra.in/",
        "tags": ["sc_st", "women", "entrepreneur", "loan"],
    },

    # ── Social Welfare & Pension ──────────────────────────────────────────────
    {
        "scheme_id": "NSAP",
        "name": "National Social Assistance Programme",
        "description": "Pension and lump sum assistance for old age persons, widows, disabled persons, and bereaved families of deceased workers.",
        "ministry": "Ministry of Rural Development",
        "category": "social_welfare",
        "benefits": "Rs 200-500/month pension; Rs 20000 lump sum on death",
        "min_age": 60, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://nsap.nic.in/",
        "tags": ["pension", "elderly", "widow", "disabled", "bpl"],
    },
    {
        "scheme_id": "PMUY",
        "name": "Pradhan Mantri Ujjwala Yojana",
        "description": "Free LPG connections to women from BPL households to provide clean cooking fuel.",
        "ministry": "Ministry of Petroleum and Natural Gas",
        "category": "social_welfare",
        "benefits": "Free LPG connection with first refill and stove free of cost",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "female", "application_url": "https://www.pmuy.gov.in/",
        "tags": ["women", "bpl", "lpg", "cooking_fuel", "free"],
    },
    {
        "scheme_id": "PM_AWAS_YOJANA_SC",
        "name": "Pradhan Mantri Adarsh Gram Yojana",
        "description": "Integrated development of SC-majority villages with convergence of central and state schemes.",
        "ministry": "Ministry of Social Justice",
        "category": "social_welfare",
        "benefits": "Infrastructure development, sanitation, road connectivity for SC villages",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://socialjustice.gov.in/",
        "tags": ["sc_st", "village_development", "infrastructure"],
    },

    # ── Women Empowerment ─────────────────────────────────────────────────────
    {
        "scheme_id": "SUKANYA",
        "name": "Sukanya Samriddhi Yojana",
        "description": "Savings scheme for girl child with high interest rate and full tax exemption on deposit, interest and maturity.",
        "ministry": "Ministry of Finance",
        "category": "women_empowerment",
        "benefits": "8.2% interest rate, tax-free maturity, minimum deposit Rs 250",
        "min_age": 0, "max_age": 10, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "female", "application_url": "https://www.nsiindia.gov.in/",
        "tags": ["girl_child", "savings", "education", "tax_free"],
    },
    {
        "scheme_id": "BETI_BACHAO",
        "name": "Beti Bachao Beti Padhao",
        "description": "Addresses declining child sex ratio and promotes girl child education with scholarships.",
        "ministry": "Ministry of Women and Child Development",
        "category": "women_empowerment",
        "benefits": "Educational scholarships, conditional cash transfers, awareness programs",
        "min_age": 0, "max_age": 18, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "female", "application_url": "https://wcd.nic.in/bbbp-schemes",
        "tags": ["girl_child", "education", "women", "scholarship"],
    },
    {
        "scheme_id": "PMMVY",
        "name": "Pradhan Mantri Matru Vandana Yojana",
        "description": "Maternity benefit of Rs 5000 to pregnant women and lactating mothers for first live birth.",
        "ministry": "Ministry of Women and Child Development",
        "category": "women_empowerment",
        "benefits": "Rs 5000 in 3 installments for first pregnancy",
        "min_age": 19, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "female", "application_url": "https://wcd.nic.in/",
        "tags": ["maternity", "women", "pregnancy", "cash_transfer"],
    },
    {
        "scheme_id": "NARI_SHAKTI",
        "name": "Nari Shakti Puraskar",
        "description": "Awards to institutions and individuals working for the cause of women empowerment.",
        "ministry": "Ministry of Women and Child Development",
        "category": "women_empowerment",
        "benefits": "Rs 2 lakh award and certificate of recognition",
        "min_age": 25, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "female", "application_url": "https://wcd.nic.in/",
        "tags": ["women", "award", "empowerment"],
    },

    # ── Education ─────────────────────────────────────────────────────────────
    {
        "scheme_id": "PM_SCHOLARSHIP",
        "name": "Prime Minister's Scholarship Scheme",
        "description": "Scholarships for dependent wards and widows of ex-servicemen/ex-Coast Guard personnel.",
        "ministry": "Ministry of Defence",
        "category": "education",
        "benefits": "Rs 2500/month for boys and Rs 3000/month for girls",
        "min_age": 18, "max_age": 25, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://ksb.gov.in/",
        "tags": ["scholarship", "education", "defence", "student"],
    },
    {
        "scheme_id": "NSP_MINORITY",
        "name": "Pre-Matric and Post-Matric Scholarship for Minorities",
        "description": "Scholarships for students from minority communities (Muslim, Christian, Sikh, Buddhist, Jain, Parsi).",
        "ministry": "Ministry of Minority Affairs",
        "category": "education",
        "benefits": "Rs 1000-12000/year for pre-matric; Rs 3000-10000/year for post-matric",
        "min_age": 6, "max_age": 32, "max_income_annual": 200000,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://scholarships.gov.in/",
        "tags": ["scholarship", "minority", "education", "student"],
    },
    {
        "scheme_id": "SC_SCHOLARSHIP",
        "name": "Post Matric Scholarship for SC Students",
        "description": "Scholarship for Scheduled Caste students pursuing post-matric level courses.",
        "ministry": "Ministry of Social Justice and Empowerment",
        "category": "education",
        "benefits": "Course fee + maintenance allowance Rs 380-1200/month",
        "min_age": 15, "max_age": None, "max_income_annual": 250000,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://scholarships.gov.in/",
        "tags": ["scholarship", "sc_st", "education", "student"],
    },
    {
        "scheme_id": "POSHAN_ABHIYAAN",
        "name": "Poshan Abhiyaan (National Nutrition Mission)",
        "description": "Improve nutritional outcomes for children, pregnant women and lactating mothers.",
        "ministry": "Ministry of Women and Child Development",
        "category": "education",
        "benefits": "Free nutritional supplements, growth monitoring, counselling",
        "min_age": 0, "max_age": 6, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://poshanabhiyaan.gov.in/",
        "tags": ["nutrition", "child", "free_service"],
    },

    # ── Health ────────────────────────────────────────────────────────────────
    {
        "scheme_id": "PMJAY",
        "name": "Ayushman Bharat PM-JAY",
        "description": "Health cover of Rs 5 lakh per family per year for secondary and tertiary hospitalisation at empanelled hospitals.",
        "ministry": "Ministry of Health and Family Welfare",
        "category": "health",
        "benefits": "Rs 5 lakh/family/year for hospitalization; cashless treatment at 25000+ hospitals",
        "min_age": 0, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://pmjay.gov.in/",
        "tags": ["health_insurance", "hospitalization", "free", "family", "cashless"],
    },
    {
        "scheme_id": "JANAUSHADHI",
        "name": "Pradhan Mantri Bhartiya Janaushadhi Pariyojana",
        "description": "Affordable generic medicines at 50-90% lower prices through Janaushadhi Kendras.",
        "ministry": "Ministry of Chemicals and Fertilizers",
        "category": "health",
        "benefits": "Generic medicines at 50-90% discount over branded medicines",
        "min_age": 0, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://janaushadhi.gov.in/",
        "tags": ["medicine", "affordable", "generic", "health"],
    },
    {
        "scheme_id": "AYUSHMAN_AROGYA",
        "name": "Ayushman Arogya Mandir (Health and Wellness Centres)",
        "description": "Comprehensive primary health care including free medicines, diagnostics and telehealth.",
        "ministry": "Ministry of Health and Family Welfare",
        "category": "health",
        "benefits": "Free medicines, free diagnostics, telehealth consultations",
        "min_age": 0, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://ab-hwc.nhp.gov.in/",
        "tags": ["primary_health", "free_medicine", "telemedicine"],
    },

    # ── Sanitation & Water ────────────────────────────────────────────────────
    {
        "scheme_id": "SBM",
        "name": "Swachh Bharat Mission - Gramin",
        "description": "Construction of individual household latrines and solid/liquid waste management in rural areas.",
        "ministry": "Ministry of Jal Shakti",
        "category": "sanitation",
        "benefits": "Rs 12000 incentive for construction of individual household toilet",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://swachhbharatmission.gov.in/",
        "tags": ["sanitation", "toilet", "rural", "hygiene"],
    },
    {
        "scheme_id": "JAL_JEEVAN",
        "name": "Jal Jeevan Mission",
        "description": "Provision of safe and adequate drinking water through individual household tap connections to every rural household.",
        "ministry": "Ministry of Jal Shakti",
        "category": "sanitation",
        "benefits": "Free tap water connection to every rural household",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://jaljeevanmission.gov.in/",
        "tags": ["water", "drinking_water", "rural", "free"],
    },

    # ── Energy ────────────────────────────────────────────────────────────────
    {
        "scheme_id": "PMUY",
        "name": "Pradhan Mantri Ujjwala Yojana",
        "description": "Free LPG connections to women from BPL households to provide clean cooking fuel and protect health.",
        "ministry": "Ministry of Petroleum and Natural Gas",
        "category": "social_welfare",
        "benefits": "Free LPG connection with first refill and stove",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "female", "application_url": "https://www.pmuy.gov.in/",
        "tags": ["women", "bpl", "lpg", "cooking_fuel"],
    },
    {
        "scheme_id": "KUSUM",
        "name": "PM KUSUM Scheme",
        "description": "Solar pumps and grid-connected solar power plants for farmers to reduce electricity and diesel costs.",
        "ministry": "Ministry of New and Renewable Energy",
        "category": "agriculture",
        "benefits": "60% subsidy on solar pump installation; Rs 6-7/unit for solar power sold to grid",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": ["farmer"],
        "gender": "all", "application_url": "https://mnre.gov.in/",
        "tags": ["solar", "farmer", "energy", "subsidy"],
    },
    {
        "scheme_id": "PM_SURYA_GHAR",
        "name": "PM Surya Ghar Muft Bijli Yojana",
        "description": "Rooftop solar installation on homes to provide 300 units free electricity per month.",
        "ministry": "Ministry of New and Renewable Energy",
        "category": "housing",
        "benefits": "Rs 30000-78000 subsidy on rooftop solar; 300 units free electricity/month",
        "min_age": 18, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://pmsuryaghar.gov.in/",
        "tags": ["solar", "electricity", "free", "subsidy", "housing"],
    },

    # ── Digital & Infrastructure ──────────────────────────────────────────────
    {
        "scheme_id": "PMGDISHA",
        "name": "Pradhan Mantri Gramin Digital Saksharta Abhiyan",
        "description": "Digital literacy to rural citizens to enable them to operate computers and digital payments.",
        "ministry": "Ministry of Electronics and IT",
        "category": "education",
        "benefits": "Free digital literacy training covering computer basics, internet, UPI payments",
        "min_age": 14, "max_age": 60, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://www.pmgdisha.in/",
        "tags": ["digital_literacy", "rural", "free_training", "internet"],
    },
    {
        "scheme_id": "COMMON_SERVICE",
        "name": "Common Service Centres Scheme",
        "description": "Single-window services for government documents, certificates, banking, insurance at village level.",
        "ministry": "Ministry of Electronics and IT",
        "category": "financial_inclusion",
        "benefits": "Access to 300+ government services at village level CSC",
        "min_age": 0, "max_age": None, "max_income_annual": None,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://www.csc.gov.in/",
        "tags": ["government_services", "village", "digital"],
    },

    # ── Disability ────────────────────────────────────────────────────────────
    {
        "scheme_id": "ASSISTANCE_DISABLED",
        "name": "Assistance to Disabled Persons for Purchase of Aids and Appliances",
        "description": "Financial assistance to disabled persons with monthly income below Rs 20000 for purchase of aids.",
        "ministry": "Ministry of Social Justice and Empowerment",
        "category": "social_welfare",
        "benefits": "Aids and appliances worth Rs 10000-1 lakh depending on disability type",
        "min_age": 0, "max_age": None, "max_income_annual": 240000,
        "eligible_states": [], "eligible_occupations": [],
        "gender": "all", "application_url": "https://socialjustice.gov.in/",
        "tags": ["disability", "aids", "financial_assistance"],
    },
    {
        "scheme_id": "NHFDC",
        "name": "National Handicapped Finance and Development Corporation",
        "description": "Concessional loans for self-employment to persons with disabilities.",
        "ministry": "Ministry of Social Justice and Empowerment",
        "category": "livelihood",
        "benefits": "Loans at 5% interest for self-employment up to Rs 30 lakh",
        "min_age": 18, "max_age": 55, "max_income_annual": 300000,
        "eligible_states": [], "eligible_occupations": ["self_employed"],
        "gender": "all", "application_url": "https://nhfdc.nic.in/",
        "tags": ["disability", "loan", "self_employment"],
    },
]

# Remove duplicates by scheme_id
seen = set()
unique_schemes = []
for s in KNOWN_SCHEMES:
    if s["scheme_id"] not in seen:
        seen.add(s["scheme_id"])
        unique_schemes.append(s)


def _fetch_from_huggingface() -> list[dict]:
    """Fetch 2066 schemes from Aryan-Pardeshi/gov-myscheme-dataset on GitHub."""
    import csv, io, requests, re

    CSV_URL = "https://raw.githubusercontent.com/Aryan-Pardeshi/gov-myscheme-dataset/main/gov_myscheme_data.csv"
    LOCAL_CACHE = Path("data/schemes/gov_myscheme_raw.csv")

    # Use cached file if available
    if LOCAL_CACHE.exists():
        print(f"Using cached dataset: {LOCAL_CACHE}")
        text = LOCAL_CACHE.read_text(encoding="utf-8-sig")
    else:
        print("Downloading gov-myscheme-dataset (15MB)...")
        try:
            r = requests.get(CSV_URL, timeout=60)
            r.raise_for_status()
            LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
            LOCAL_CACHE.write_bytes(r.content)
            text = r.content.decode("utf-8-sig")
            print(f"Downloaded {len(r.content)//1024}KB")
        except Exception as e:
            print(f"[GitHub] Download failed: {e}")
            return []

    reader = csv.DictReader(io.StringIO(text))
    records = []
    for i, row in enumerate(reader):
        name = row.get("Scheme Name", "").strip()
        if not name:
            continue

        slug = row.get("Scheme Slug", "").strip().upper().replace("-", "_")
        ministry = row.get("State / UT / Ministry", "").strip()
        tags_raw = row.get("Tags / Categories", "")
        tags = [t.strip().lower() for t in re.split(r"[,;|]", tags_raw) if t.strip()]
        eligibility = row.get("Eligibility Criteria", "") or row.get("Eligibility (General)", "")

        records.append({
            "scheme_id": slug or f"MS_{i:04d}",
            "name": name,
            "description": row.get("Description", "").strip(),
            "ministry": ministry,
            "category": tags[0] if tags else "general",
            "benefits": row.get("Benefits", "").strip(),
            "min_age": None,
            "max_age": None,
            "max_income_annual": None,
            "eligible_states": [],
            "eligible_occupations": [],
            "gender": "all",
            "application_url": row.get("Official Link", "").strip() or row.get("MyScheme URL", "").strip(),
            "tags": tags,
            "eligibility_text": eligibility.strip(),
            "documents_required": row.get("Documents Required", "").strip(),
            "source": "myscheme.gov.in",
        })

    print(f"✓ Parsed {len(records)} schemes from myScheme dataset")
    return records


def run():
    hf_records = _fetch_from_huggingface()

    if hf_records:
        # Use HuggingFace dataset — 2066 schemes
        all_records = hf_records
    else:
        # Fallback to hand-curated
        all_records = unique_schemes

    # Deduplicate
    seen = set()
    final = []
    for r in all_records:
        key = r.get("scheme_id") or r.get("name", "")
        if key and key not in seen:
            seen.add(key)
            final.append(r)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for scheme in final:
            f.write(json.dumps(scheme, ensure_ascii=False) + "\n")
    print(f"✓ Saved {len(final)} schemes to {OUT_FILE}")


if __name__ == "__main__":
    run()
