"""
Build FAISS index and artifacts for the insurance recommender.
Member B owns this file.

Run: python models/insurance_recommender/train.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_PATH    = Path("data/insurance/insurance_clean.jsonl")
ARTIFACTS    = Path("models/insurance_recommender/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

EMBED_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def run():
    records = [
        json.loads(l) for l in DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    print(f"Loaded {len(records)} insurance records")

    model = SentenceTransformer(EMBED_MODEL)
    texts = [r.get("embed_text", r.get("name", "")) for r in records]

    print("Building embeddings...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(ARTIFACTS / "insurance_faiss.index"))
    np.save(ARTIFACTS / "insurance_embeddings.npy", embeddings)
    with open(ARTIFACTS / "insurance_records.pkl", "wb") as f:
        pickle.dump(records, f)

    print(f"✓ Insurance recommender index built ({len(records)} items, {dim}d)")


if __name__ == "__main__":
    run()
