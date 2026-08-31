"""
MEMBER A — Step 1: Collect government schemes data.

Sources (in priority order):
  1. Hugging Face: shrijayan/gov_myscheme  (723 PDFs already parsed, CSV/JSON/Parquet)
  2. myScheme public search API            (live, structured JSON)
  3. HTML scrape fallback                  (if API rate-limits)

Output: data/schemes/raw_schemes.jsonl
"""

import json
import time
from pathlib import Path

import httpx
from tqdm import tqdm

OUT_DIR = Path(__file__).parent.parent / "schemes"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "raw_schemes.jsonl"

MYSCHEME_SEARCH_API = "https://api.myscheme.gov.in/search/v4/schemes"
HF_DATASET = "shrijayan/gov_myscheme"
PAGE_SIZE = 100


def fetch_from_hf() -> list[dict]:
    """
    Download the pre-built HuggingFace dataset (fastest, already cleaned).
    Run once: `python -c "from datasets import load_dataset; ds = load_dataset('shrijayan/gov_myscheme'); ds['train'].to_json('data/schemes/raw_schemes.jsonl')"` 
    This function does the same programmatically.
    """
    print("[HF] Loading shrijayan/gov_myscheme from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_DATASET, split="train")
        records = [dict(row) for row in ds]
        print(f"[HF] Loaded {len(records)} records from HuggingFace dataset")
        return records
    except Exception as e:
        print(f"[HF] Failed: {e}")
        return []


def fetch_from_api() -> list[dict]:
    """Live myScheme API — returns structured scheme data."""
    print("[API] Fetching from myScheme API...")
    records = []
    offset = 0
    with httpx.Client(timeout=30) as client:
        while True:
            try:
                resp = client.get(MYSCHEME_SEARCH_API, params={"from": offset, "size": PAGE_SIZE})
                resp.raise_for_status()
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                if not hits:
                    break
                records.extend(hits)
                print(f"[API] Fetched {len(records)} so far...")
                offset += PAGE_SIZE
                time.sleep(0.5)  # be polite
            except Exception as e:
                print(f"[API] Error at offset {offset}: {e}")
                break
    print(f"[API] Total from API: {len(records)}")
    return records


def normalize_hf_record(r: dict) -> dict:
    """Normalize HuggingFace dataset row to our schema."""
    return {
        "scheme_id": r.get("scheme_id") or r.get("id") or "",
        "name": r.get("scheme_name") or r.get("name") or "",
        "description": r.get("description") or r.get("short_description") or "",
        "ministry": r.get("ministry") or r.get("nodal_ministry") or "",
        "category": r.get("category") or "",
        "eligibility": r.get("eligibility") or [],
        "benefits": r.get("benefits") or "",
        "application_process": r.get("application_process") or "",
        "documents_required": r.get("documents_required") or [],
        "min_age": _safe_int(r.get("min_age")),
        "max_age": _safe_int(r.get("max_age")),
        "max_income_annual": _safe_float(r.get("max_income") or r.get("income_limit")),
        "eligible_states": r.get("states") or [],
        "eligible_occupations": r.get("occupation") or r.get("target_beneficiaries") or [],
        "gender": r.get("gender") or "all",
        "application_url": r.get("application_url") or r.get("url") or "",
        "source": "huggingface/gov_myscheme",
    }


def normalize_api_record(r: dict) -> dict:
    """Normalize myScheme API hit to our schema."""
    src = r.get("_source", {})
    return {
        "scheme_id": r.get("_id", ""),
        "name": src.get("schemeName", ""),
        "description": src.get("schemeShortTitle", "") or src.get("schemeName", ""),
        "ministry": src.get("ministryName", ""),
        "category": src.get("schemeCategory", ""),
        "eligibility": src.get("eligibility", []),
        "benefits": src.get("benefit", ""),
        "application_process": src.get("applicationProcess", ""),
        "documents_required": src.get("documents", []),
        "min_age": _safe_int(src.get("minAge")),
        "max_age": _safe_int(src.get("maxAge")),
        "max_income_annual": _safe_float(src.get("incomeLimit")),
        "eligible_states": src.get("state", []),
        "eligible_occupations": src.get("beneficiary", []),
        "gender": src.get("gender", "all"),
        "application_url": src.get("applicationUrl", ""),
        "source": "myscheme_api",
    }


def _safe_int(v) -> int | None:
    try:
        return int(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _safe_float(v) -> float | None:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def run():
    # Try HuggingFace first (best quality, pre-parsed)
    raw = fetch_from_hf()

    if not raw:
        print("[fallback] HF unavailable, trying live API...")
        raw = fetch_from_api()

    if not raw:
        print("[ERROR] No data collected. Run generate_synthetic_schemes.py as fallback.")
        return

    # Normalize
    normalized = []
    for r in tqdm(raw, desc="Normalizing"):
        if r.get("_source"):
            normalized.append(normalize_api_record(r))
        else:
            normalized.append(normalize_hf_record(r))

    # Write
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for rec in normalized:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✓ Saved {len(normalized)} scheme records to {OUT_FILE}")


if __name__ == "__main__":
    run()
