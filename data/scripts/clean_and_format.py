"""
Step 2 (both members): Clean and format raw collected data into a unified schema.
Merges schemes + insurance into their respective cleaned files.

Input:  data/schemes/raw_schemes.jsonl
        data/insurance/raw_insurance.jsonl
        data/schemes/synthetic_schemes.jsonl  (optional)

Output: data/schemes/schemes_clean.jsonl
        data/insurance/insurance_clean.jsonl
        data/training/all_items.jsonl         (merged, for FAISS / recommender)
"""

import json
import re
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[WARN] {path} not found, skipping")
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def clean_scheme(r: dict) -> dict | None:
    name = str(r.get("name", "")).strip()
    if not name or name.lower() in ("", "none", "nan"):
        return None  # drop blank records

    return {
        "id": str(r.get("scheme_id", "")).strip(),
        "type": "scheme",
        "name": name,
        "description": _clean_text(r.get("description", "")),
        "ministry": _clean_text(r.get("ministry", "")),
        "category": str(r.get("category", "")).strip().lower(),
        "benefits": _clean_text(r.get("benefits", "")),
        # Eligibility — normalised to consistent types
        "min_age": _safe_int(r.get("min_age")),
        "max_age": _safe_int(r.get("max_age")),
        "max_income_annual": _safe_float(r.get("max_income_annual") or r.get("max_income")),
        "eligible_states": _normalise_list(r.get("eligible_states")),
        "eligible_occupations": _normalise_list(r.get("eligible_occupations") or r.get("eligibility")),
        "gender": str(r.get("gender", "all")).lower(),
        "application_url": str(r.get("application_url", "")).strip(),
        "tags": _normalise_list(r.get("tags")),
        "source": r.get("source", "unknown"),
        # Text for embedding (used by recommender and FAISS)
        "embed_text": f"{name}. {r.get('description', '')}. {r.get('benefits', '')}",
    }


def clean_insurance(r: dict) -> dict | None:
    name = str(r.get("name", "")).strip()
    if not name:
        return None
    return {
        "id": str(r.get("scheme_id", "")).strip(),
        "type": "insurance",
        "name": name,
        "insurance_type": str(r.get("type", "")).strip().lower(),
        "description": _clean_text(r.get("description", "")),
        "insurer": str(r.get("insurer", "")).strip(),
        "premium_annual": _safe_float(r.get("premium_annual")),
        "sum_assured": _safe_float(r.get("sum_assured")),
        "benefits": _clean_text(r.get("benefits", "")),
        "min_age": _safe_int(r.get("min_age")),
        "max_age": _safe_int(r.get("max_age")),
        "max_income_annual": _safe_float(r.get("max_income_annual")),
        "eligible_states": _normalise_list(r.get("eligible_states")),
        "eligible_occupations": _normalise_list(r.get("eligible_occupations")),
        "gender": str(r.get("gender", "all")).lower(),
        "bank_account_required": bool(r.get("bank_account_required", False)),
        "aadhaar_required": bool(r.get("aadhaar_required", False)),
        "application_url": str(r.get("application_url", "")).strip(),
        "tags": _normalise_list(r.get("tags")),
        "source": r.get("source", "unknown"),
        "embed_text": f"{name}. {r.get('description', '')}. {r.get('benefits', '')}",
    }


def _clean_text(v) -> str:
    if not v:
        return ""
    text = str(v).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:2000]  # cap length


def _safe_int(v) -> int | None:
    try:
        return int(float(str(v).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


def _safe_float(v) -> float | None:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _normalise_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip().lower() for x in v if x]
    if isinstance(v, str) and v:
        return [x.strip().lower() for x in re.split(r"[,;|]", v) if x.strip()]
    return []


def run():
    # ── Schemes ──────────────────────────────────────────────────────────────
    raw_schemes = load_jsonl(DATA_DIR / "schemes/raw_schemes.jsonl")
    synth_schemes = load_jsonl(DATA_DIR / "schemes/synthetic_schemes.jsonl")
    all_raw_schemes = raw_schemes + synth_schemes

    cleaned_schemes = []
    for r in tqdm(all_raw_schemes, desc="Cleaning schemes"):
        cleaned = clean_scheme(r)
        if cleaned:
            cleaned_schemes.append(cleaned)

    # Deduplicate by id
    seen = set()
    deduped_schemes = []
    for s in cleaned_schemes:
        key = s["id"] or s["name"]
        if key not in seen:
            seen.add(key)
            deduped_schemes.append(s)

    out_schemes = DATA_DIR / "schemes/schemes_clean.jsonl"
    with open(out_schemes, "w", encoding="utf-8") as f:
        for s in deduped_schemes:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✓ {len(deduped_schemes)} clean schemes → {out_schemes}")

    # ── Insurance ─────────────────────────────────────────────────────────────
    raw_insurance = load_jsonl(DATA_DIR / "insurance/raw_insurance.jsonl")
    synth_insurance = load_jsonl(DATA_DIR / "insurance/synthetic_insurance.jsonl")
    all_raw_insurance = raw_insurance + synth_insurance

    cleaned_insurance = []
    for r in tqdm(all_raw_insurance, desc="Cleaning insurance"):
        cleaned = clean_insurance(r)
        if cleaned:
            cleaned_insurance.append(cleaned)

    out_insurance = DATA_DIR / "insurance/insurance_clean.jsonl"
    with open(out_insurance, "w", encoding="utf-8") as f:
        for s in cleaned_insurance:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✓ {len(cleaned_insurance)} clean insurance records → {out_insurance}")

    # ── Merged for recommender ─────────────────────────────────────────────
    all_items = deduped_schemes + cleaned_insurance
    out_all = DATA_DIR / "training/all_items.jsonl"
    out_all.parent.mkdir(parents=True, exist_ok=True)
    with open(out_all, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ {len(all_items)} total items merged → {out_all}")


if __name__ == "__main__":
    run()
