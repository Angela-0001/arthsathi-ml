"""
ArthSathi Language Model API — self-hosted, no external APIs.
Serves our own trained GPT-style model for scheme Q&A.

Run: uvicorn language_model.serve:app --port 5002 --reload
"""
import sys
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(
    title="ArthSathi LM API",
    description="Self-hosted language model for scheme and financial Q&A.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7
    top_k: int = 40


class QueryResponse(BaseModel):
    response: str
    model: str = "arthsathi-lm-v1"


@lru_cache(maxsize=1)
def get_model():
    import torch
    from language_model.model import ArthSathiLM
    from translation.tokenizer import ArthSathiTokenizer

    ckpt_path = Path("language_model/checkpoints/best_lm.pt")
    if not ckpt_path.exists():
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    tok = ArthSathiTokenizer()
    model = ArthSathiLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        max_len=cfg["max_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok


@app.get("/health")
def health():
    return {"status": "ok", "service": "arthsathi-lm"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    import torch

    model, tok = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run: python language_model/train.py"
        )

    prompt = f"### Question: {req.prompt}\n### Answer:"
    ids = tok.encode(prompt, max_length=256)
    input_ids = torch.tensor([ids])

    output = model.generate(
        input_ids,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    generated = tok.decode(output[0].tolist()[len(ids):])
    return QueryResponse(response=generated.strip())
