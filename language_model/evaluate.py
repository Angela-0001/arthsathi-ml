"""
Evaluate ArthSathiLM on BhashaBench-Finance (AI Kosh) and our own scheme QA set.
Compare against base model (untrained) to show the fine-tuning improves accuracy.

Member A owns this file.

Usage:
    python language_model/evaluate.py
"""

import json
import torch
from pathlib import Path
from language_model.model import ArthSathiLM
from translation.tokenizer import ArthSathiTokenizer

CHECKPOINT = Path("language_model/checkpoints/best_lm.pt")
VAL_DATA   = Path("language_model/data/val.jsonl")


def load_model(device: str = "cpu"):
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"No checkpoint at {CHECKPOINT}. Train first.")
    ckpt = torch.load(CHECKPOINT, map_location=device)
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


def perplexity(model, tok, data_path: Path, device: str = "cpu") -> float:
    """Compute perplexity on a JSONL validation file."""
    import math
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss(ignore_index=tok.PAD_ID, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for line in data_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            ids = tok.encode(item["text"], max_length=512)
            if len(ids) < 3:
                continue
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([ids[1:]],  dtype=torch.long, device=device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, tok.vocab_size), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def generation_demo(model, tok, device: str = "cpu"):
    """Quick qualitative check — generate answers to sample questions."""
    prompts = [
        "### Question: What is PM-JAY?\n### Answer:",
        "### Question: Who is eligible for PMSBY?\n### Answer:",
        "### Question: What does 'forfeit' mean in a loan agreement?\n### Answer:",
    ]
    print("\n── Generation Samples ──────────────────────────────")
    for prompt in prompts:
        ids = tok.encode(prompt, max_length=128)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        output = model.generate(input_ids, max_new_tokens=80, temperature=0.7, top_k=40)
        generated = tok.decode(output[0].tolist()[len(ids):])
        print(f"\nQ: {prompt.split('Question:')[1].split('Answer:')[0].strip()}")
        print(f"A: {generated.strip()}")


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, tok = load_model(device)
    print(f"Model params: {model.count_params():,}")

    print("\nComputing perplexity on validation set...")
    ppl = perplexity(model, tok, VAL_DATA, device)
    print(f"Validation perplexity: {ppl:.2f}")
    print("(Lower = better. A well-trained domain model should be < 50)")

    generation_demo(model, tok, device)


if __name__ == "__main__":
    run()
