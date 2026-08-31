"""
Build the instruction-tuning dataset for ArthSathiLM.

Format: instruction-following QA pairs built from scheme + insurance records.
Each record becomes multiple training examples:
  - Eligibility check: "Is a 45 year old farmer from Maharashtra eligible for PMFBY?"
  - Benefit query: "What does PM-JAY provide?"
  - Document explanation: "What does 'irrevocable' mean in a loan agreement?"

Output: language_model/data/train.jsonl, val.jsonl
Each line: {"prompt": "...", "completion": "..."}

Member A owns this file.
"""

import json
import random
from pathlib import Path

random.seed(42)

SCHEMES_FILE = Path("data/schemes/schemes_clean.jsonl")
INSURANCE_FILE = Path("data/insurance/insurance_clean.jsonl")
OUT_DIR = Path("language_model/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Legal/financial term explanations — hand-curated (this IS original work)
LEGAL_TERMS = [
    ("irrevocable", "एक बार सहमत होने के बाद इसे रद्द नहीं किया जा सकता।",
     "Once agreed, it cannot be cancelled or undone."),
    ("forfeit", "अगर शर्त पूरी नहीं हुई तो आप अपना पैसा या संपत्ति खो देंगे।",
     "You lose your money or property if a condition is not met."),
    ("collateral", "कर्ज लेते समय गारंटी के रूप में दी गई संपत्ति।",
     "Property given as guarantee when taking a loan."),
    ("compound interest", "ब्याज पर भी ब्याज लगता है — कर्ज तेजी से बढ़ता है।",
     "Interest charged on interest — debt grows faster over time."),
    ("penalty clause", "देर से भुगतान या शर्त तोड़ने पर अतिरिक्त जुर्माना।",
     "Extra charge for late payment or breaking a condition."),
    ("arbitration only", "विवाद होने पर केवल मध्यस्थ के पास जा सकते हैं, अदालत नहीं।",
     "If there is a dispute, you can only go to an arbitrator, not a court."),
    ("automatic renewal", "अनुबंध अपने आप अगले साल के लिए नवीनीकृत हो जाएगा।",
     "The contract will automatically extend for another year."),
    ("unlimited liability", "आप कंपनी के सभी कर्ज के लिए व्यक्तिगत रूप से जिम्मेदार हैं।",
     "You are personally responsible for all debts of the company."),
]


def load_jsonl(path: Path) -> list:
    if not path.exists():
        print(f"[WARN] {path} not found")
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def scheme_to_examples(s: dict) -> list[dict]:
    examples = []
    name = s.get("name", "")
    desc = s.get("description", "") or s.get("benefits", "")
    min_age = s.get("min_age")
    max_age = s.get("max_age")
    max_income = s.get("max_income_annual")
    occs = s.get("eligible_occupations", [])
    states = s.get("eligible_states", [])

    # Q1: What is this scheme?
    examples.append({
        "prompt": f"What is {name}?",
        "completion": f"{name} is a government scheme. {desc}",
    })

    # Q2: Who is eligible?
    elig_parts = []
    if min_age:
        elig_parts.append(f"age at least {min_age}")
    if max_age:
        elig_parts.append(f"age at most {max_age}")
    if max_income:
        elig_parts.append(f"annual income below ₹{max_income:,.0f}")
    if occs:
        elig_parts.append(f"occupation: {', '.join(occs[:3])}")
    if states:
        elig_parts.append(f"resident of: {', '.join(states[:3])}")

    if elig_parts:
        examples.append({
            "prompt": f"Who is eligible for {name}?",
            "completion": f"To be eligible for {name}, you must meet these criteria: {'; '.join(elig_parts)}.",
        })

    # Q3: Eligibility check with a user profile
    if min_age and max_income:
        age_ok = min_age + 5
        income_ok = int(max_income * 0.6)
        examples.append({
            "prompt": f"I am {age_ok} years old with annual income ₹{income_ok:,}. Am I eligible for {name}?",
            "completion": f"Yes, based on the criteria for {name}, you appear eligible. Your age ({age_ok}) meets the minimum age requirement ({min_age}), and your income (₹{income_ok:,}) is within the limit.",
        })

    return examples


def insurance_to_examples(ins: dict) -> list[dict]:
    examples = []
    name = ins.get("name", "")
    ins_type = ins.get("insurance_type", "")
    benefits = ins.get("benefits", "")
    premium = ins.get("premium_annual")
    sum_assured = ins.get("sum_assured")

    examples.append({
        "prompt": f"What is {name}?",
        "completion": f"{name} is a {ins_type} insurance scheme. {benefits}",
    })

    if premium is not None and sum_assured:
        examples.append({
            "prompt": f"How much does {name} cost and what does it cover?",
            "completion": f"{name} costs ₹{premium:,.0f} per year and provides coverage of ₹{sum_assured:,.0f}. {benefits}",
        })

    return examples


def legal_term_examples() -> list[dict]:
    examples = []
    for term, hindi_exp, english_exp in LEGAL_TERMS:
        examples.append({
            "prompt": f"What does '{term}' mean in a loan agreement?",
            "completion": english_exp,
        })
        examples.append({
            "prompt": f"'{term}' का मतलब क्या है?",
            "completion": hindi_exp,
        })
    return examples


def run():
    schemes = load_jsonl(SCHEMES_FILE)
    insurance = load_jsonl(INSURANCE_FILE)

    all_examples = []

    for s in schemes:
        all_examples.extend(scheme_to_examples(s))

    for ins in insurance:
        all_examples.extend(insurance_to_examples(ins))

    all_examples.extend(legal_term_examples())

    random.shuffle(all_examples)

    # Convert to instruction format: "<prompt>\n<completion>"
    # We use a simple text format — the model learns to complete after the newline
    formatted = []
    for ex in all_examples:
        formatted.append({
            "text": f"### Question: {ex['prompt']}\n### Answer: {ex['completion']}"
        })

    # Train/val split
    val_size = max(100, int(len(formatted) * 0.05))
    val = formatted[:val_size]
    train = formatted[val_size:]

    with open(OUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(OUT_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✓ {len(train)} training examples → language_model/data/train.jsonl")
    print(f"✓ {len(val)} validation examples → language_model/data/val.jsonl")


if __name__ == "__main__":
    run()
