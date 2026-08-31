"""
MEMBER A — Scheme Recommendation Model

Architecture:
  - Eligibility filter  : hard rule-based pass (deterministic, explainable)
  - Content embedding   : sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
                          embeds scheme descriptions into a vector space
  - User profile vector : encodes user's profile into the same space
  - Ranking             : cosine similarity + adaptive weight multiplier
  - Adaptive weights    : updated by AdaptiveWeightEngine (shared module)

Training = building the FAISS index over scheme embeddings.
No gradient training needed for the base recommender — the "learning" happens
through the adaptive weight updates at inference time.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_PATH = Path("data/schemes/schemes_clean.jsonl")
MODEL_OUT = Path("models/scheme_recommender/artifacts")
MODEL_OUT.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Multilingual: supports Hindi, Marathi, English — 50MB, runs on CPU


def load_schemes() -> list[dict]:
    return [json.loads(l) for l in DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_index(schemes: list[dict], model: SentenceTransformer):
    import faiss

    texts = [s["embed_text"] for s in schemes]
    print(f"Embedding {len(texts)} schemes...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine sim (since normalized)
    index.add(embeddings)

    # Save
    faiss.write_index(index, str(MODEL_OUT / "scheme_faiss.index"))
    with open(MODEL_OUT / "scheme_records.pkl", "wb") as f:
        pickle.dump(schemes, f)
    np.save(MODEL_OUT / "scheme_embeddings.npy", embeddings)

    print(f"✓ FAISS index ({dim}d, {len(texts)} vectors) saved to {MODEL_OUT}")


def run():
    schemes = load_schemes()
    print(f"Loaded {len(schemes)} schemes")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    build_index(schemes, model)


if __name__ == "__main__":
    run()
