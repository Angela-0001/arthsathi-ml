"""
Inference: greedy decode + beam search for our trained translation model.
Member C owns this file.
"""

import torch
from pathlib import Path
from translation.model import Seq2SeqTranslator
from translation.tokenizer import ArthSathiTokenizer

CHECKPOINT_DIR = Path("translation/checkpoints")


def load_model(lang: str = "hi", device: str = "cpu"):
    ckpt_path = CHECKPOINT_DIR / f"best_{lang}_en.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first.")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    tok = ArthSathiTokenizer()
    model = Seq2SeqTranslator(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_enc_layers=cfg["n_layers"],
        n_dec_layers=cfg["n_layers"],
        max_seq_len=cfg["max_len"],
        pad_idx=tok.PAD_ID,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok


def greedy_decode(model, tok, text: str, src_lang: str, tgt_lang: str,
                  max_len: int = 256, device: str = "cpu") -> str:
    src_ids = tok.encode(text, lang=src_lang, max_length=max_len)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = Seq2SeqTranslator.make_pad_mask(src, tok.PAD_ID).to(device)

    with torch.no_grad():
        enc_out = model.encode(src, src_mask)

    # Start with BOS token
    tgt_ids = [tok.BOS_ID]

    for _ in range(max_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        tgt_mask = (
            Seq2SeqTranslator.make_pad_mask(tgt, tok.PAD_ID) &
            Seq2SeqTranslator.make_causal_mask(tgt.size(1), device)
        ).to(device)

        with torch.no_grad():
            dec_out = model.decode(tgt, enc_out, src_mask, tgt_mask)
            logits = model.output_proj(dec_out[:, -1, :])  # last token
            next_id = logits.argmax(-1).item()

        if next_id == tok.EOS_ID:
            break
        tgt_ids.append(next_id)

    return tok.decode(tgt_ids[1:])  # strip BOS


def beam_decode(model, tok, text: str, src_lang: str, tgt_lang: str,
                beam_size: int = 4, max_len: int = 256, device: str = "cpu") -> str:
    """Beam search — better quality than greedy, worth it even on CPU for inference."""
    import math

    src_ids = tok.encode(text, lang=src_lang, max_length=max_len)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = Seq2SeqTranslator.make_pad_mask(src, tok.PAD_ID).to(device)

    with torch.no_grad():
        enc_out = model.encode(src, src_mask)  # (1, S, d)

    # Each beam: (score, token_ids)
    beams = [(0.0, [tok.BOS_ID])]
    completed = []

    for _ in range(max_len):
        candidates = []
        for score, ids in beams:
            if ids[-1] == tok.EOS_ID:
                completed.append((score, ids))
                continue
            tgt = torch.tensor([ids], dtype=torch.long, device=device)
            tgt_mask = (
                Seq2SeqTranslator.make_pad_mask(tgt, tok.PAD_ID) &
                Seq2SeqTranslator.make_causal_mask(tgt.size(1), device)
            ).to(device)
            with torch.no_grad():
                dec_out = model.decode(tgt, enc_out, src_mask, tgt_mask)
                log_probs = torch.log_softmax(model.output_proj(dec_out[:, -1, :]), dim=-1)

            top_probs, top_ids = log_probs[0].topk(beam_size)
            for prob, tid in zip(top_probs.tolist(), top_ids.tolist()):
                candidates.append((score + prob, ids + [tid]))

        if not candidates:
            break

        # Keep top beam_size
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

        if len(completed) >= beam_size:
            break

    if not completed:
        completed = beams

    # Pick best by length-normalised score
    best_score, best_ids = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
    # Strip BOS/EOS
    result_ids = [i for i in best_ids if i not in (tok.BOS_ID, tok.EOS_ID)]
    return tok.decode(result_ids)


class Translator:
    """Simple interface used by the API server."""

    def __init__(self, device: str = "cpu"):
        self._models: dict = {}
        self._tok = ArthSathiTokenizer()
        self._device = device

    def _get_model(self, lang: str):
        if lang not in self._models:
            model, _ = load_model(lang, self._device)
            self._models[lang] = model
        return self._models[lang]

    def translate(self, text: str, src_lang: str, tgt_lang: str, beam: bool = True) -> str:
        # We train hi↔en and mr↔en — derive the right lang key
        pivot_lang = "hi" if "hi" in (src_lang, tgt_lang) else "mr"
        model = self._get_model(pivot_lang)
        if beam:
            return beam_decode(model, self._tok, text, src_lang, tgt_lang,
                               device=self._device)
        return greedy_decode(model, self._tok, text, src_lang, tgt_lang,
                             device=self._device)
