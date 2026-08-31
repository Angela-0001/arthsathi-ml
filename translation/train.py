"""
Train our seq2seq translation model from scratch.
Designed for Colab free tier (T4 GPU) or CPU (slower but works).

Member C owns this file.

Usage:
    python translation/train.py --lang hi --epochs 10 --batch_size 32
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from translation.model import Seq2SeqTranslator
from translation.tokenizer import ArthSathiTokenizer
from translation.dataset import TranslationDataset, collate_fn, load_pairs

CHECKPOINT_DIR = Path("translation/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ──────────────────────────────────────────────────────
    tok = ArthSathiTokenizer()
    pad_id = tok.PAD_ID

    # ── Data ───────────────────────────────────────────────────────────
    pairs = load_pairs(args.lang)
    print(f"Loaded {len(pairs)} sentence pairs for {args.lang}-en")

    dataset = TranslationDataset(pairs, tok, max_len=args.max_len)
    val_size = min(5000, int(len(dataset) * 0.05))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    pad_collate = lambda b: collate_fn(b, pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=pad_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=pad_collate, num_workers=0)

    # ── Model ──────────────────────────────────────────────────────────
    model = Seq2SeqTranslator(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_enc_layers=args.n_layers,
        n_dec_layers=args.n_layers,
        d_ff=args.d_model * 4,
        max_seq_len=args.max_len,
        pad_idx=pad_id,
    ).to(device)

    print(f"Model parameters: {model.count_params():,}")

    # ── Optimizer + scheduler ──────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # Warmup schedule (standard for transformers)
    warmup_steps = 4000
    def lr_lambda(step):
        step = max(step, 1)
        return min(step ** -0.5, step * warmup_steps ** -1.5) * (args.d_model ** -0.5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)

    # ── Training loop ──────────────────────────────────────────────────
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        t0 = time.time()

        for batch in train_loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            # Decoder input: all tokens except last
            # Decoder target: all tokens except first (shifted right)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = Seq2SeqTranslator.make_pad_mask(src, pad_id).to(device)
            tgt_mask = (
                Seq2SeqTranslator.make_pad_mask(tgt_in, pad_id) &
                Seq2SeqTranslator.make_causal_mask(tgt_in.size(1), device)
            ).to(device)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            # logits: (B, T, vocab) → need (B*T, vocab)
            loss = criterion(
                logits.reshape(-1, tok.vocab_size),
                tgt_out.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

        avg_train = total_loss / len(train_loader)
        val_loss = _evaluate(model, val_loader, criterion, device, tok.vocab_size, pad_id)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d} | train_loss={avg_train:.4f} | val_loss={val_loss:.4f} | {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = CHECKPOINT_DIR / f"best_{args.lang}_en.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "vocab_size": tok.vocab_size,
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "max_len": args.max_len,
                    "lang": args.lang,
                },
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint → {ckpt_path}")

    print(f"\nTraining done. Best val loss: {best_val_loss:.4f}")


def _evaluate(model, loader, criterion, device, vocab_size, pad_id):
    model.eval()
    total = 0
    with torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            src_mask = Seq2SeqTranslator.make_pad_mask(src, pad_id).to(device)
            tgt_mask = (
                Seq2SeqTranslator.make_pad_mask(tgt_in, pad_id) &
                Seq2SeqTranslator.make_causal_mask(tgt_in.size(1), device)
            ).to(device)
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            total += loss.item()
    return total / len(loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="hi", choices=["hi", "mr"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0)
    train(parser.parse_args())
