"""
Train ArthSathiLM on scheme + insurance + legal domain data.
Designed for Colab free tier or CPU.

Member A owns this file.

Usage:
    python language_model/train.py --epochs 5
    python language_model/train.py --epochs 10 --d_model 768 --n_layers 12  # bigger, needs GPU
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from language_model.model import ArthSathiLM
from translation.tokenizer import ArthSathiTokenizer  # reuse the same tokenizer

CHECKPOINT_DIR = Path("language_model/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("language_model/data")


class SchemeQADataset(Dataset):
    def __init__(self, path: Path, tokenizer: ArthSathiTokenizer, max_len: int = 512):
        self.examples = []
        self.tok = tokenizer
        self.max_len = max_len

        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            ids = tokenizer.encode(item["text"], max_length=max_len)
            if len(ids) > 4:  # skip very short
                self.examples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch, pad_id: int):
    max_len = max(x.size(0) for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, : x.size(0)] = x
    return padded


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok = ArthSathiTokenizer()
    pad_id = tok.PAD_ID

    train_ds = SchemeQADataset(DATA_DIR / "train.jsonl", tok, args.max_len)
    val_ds = SchemeQADataset(DATA_DIR / "val.jsonl", tok, args.max_len)

    pad_collate = lambda b: collate(b, pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=pad_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=pad_collate, num_workers=0)

    model = ArthSathiLM(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.max_len,
        pad_idx=pad_id,
    ).to(device)

    print(f"Model parameters: {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # Simple cosine LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        t0 = time.time()

        for batch in train_loader:
            batch = batch.to(device)
            # Input: all tokens except last; target: all tokens except first
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            loss = criterion(logits.reshape(-1, tok.vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x, y = batch[:, :-1], batch[:, 1:]
                logits = model(x)
                val_total += criterion(logits.reshape(-1, tok.vocab_size), y.reshape(-1)).item()
        avg_val = val_total / len(val_loader)

        print(f"Epoch {epoch:02d} | train={avg_train:.4f} | val={avg_val:.4f} | {time.time()-t0:.0f}s")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": avg_val,
                "config": {
                    "vocab_size": tok.vocab_size,
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "max_len": args.max_len,
                },
            }, CHECKPOINT_DIR / "best_lm.pt")
            print(f"  ✓ Checkpoint saved")

    print(f"\nDone. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=256)   # 256=CPU, 768=GPU
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    train(parser.parse_args())
