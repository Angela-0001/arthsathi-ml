"""
Build parallel corpus and PyTorch Dataset for training our translation model.

Data sources (all from AI Kosh / open):
  - Samanantar: English↔Hindi, English↔Marathi parallel sentences
  - PIB Multilingual Corpus: government press releases (financial/scheme domain)
  - Our own myScheme scraped pairs (scheme description bilingual pages)

Member C owns this file.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from translation.tokenizer import ArthSathiTokenizer

CORPUS_DIR = Path("translation/corpus")
MAX_LEN = 256


def download_samanantar(lang_pair: str = "hi", split: str = "train", max_samples: int = 100000):
    """
    Load Samanantar from HuggingFace datasets (AI Kosh / AI4Bharat).
    lang_pair: 'hi' (Hindi) or 'mr' (Marathi)
    Saves to translation/corpus/{lang_pair}_en.jsonl
    """
    from datasets import load_dataset

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = CORPUS_DIR / f"{lang_pair}_en.jsonl"

    if out_file.exists():
        print(f"[corpus] {out_file} already exists, skipping download")
        return

    print(f"[corpus] Downloading Samanantar {lang_pair}-en (up to {max_samples} pairs)...")
    ds = load_dataset("ai4bharat/samanantar", lang_pair, split=split, streaming=True)

    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for item in tqdm(ds, total=max_samples):
            if count >= max_samples:
                break
            pair = {
                "src": item["src"],
                "tgt": item["tgt"],
                "src_lang": "en",
                "tgt_lang": lang_pair,
            }
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            # Also write reverse direction
            reverse = {"src": item["tgt"], "tgt": item["src"],
                       "src_lang": lang_pair, "tgt_lang": "en"}
            f.write(json.dumps(reverse, ensure_ascii=False) + "\n")
            count += 1

    print(f"✓ Saved {count * 2} sentence pairs to {out_file}")


def build_plain_text_corpus() -> str:
    """
    Merge all JSONL parallel files into one plain text file for tokenizer training.
    Returns path to the plain text file.
    """
    all_sentences = []
    for jsonl_file in CORPUS_DIR.glob("*.jsonl"):
        for line in jsonl_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            all_sentences.append(item.get("src", ""))
            all_sentences.append(item.get("tgt", ""))

    # Shuffle so tokenizer sees all languages mixed
    random.shuffle(all_sentences)
    out_path = CORPUS_DIR / "all_sentences.txt"
    out_path.write_text("\n".join(all_sentences), encoding="utf-8")
    print(f"✓ Plain text corpus: {len(all_sentences)} sentences → {out_path}")
    return str(out_path)


def load_pairs(lang_pair: str = "hi") -> List[Tuple[str, str, str, str]]:
    """Returns list of (src_text, tgt_text, src_lang, tgt_lang)."""
    path = CORPUS_DIR / f"{lang_pair}_en.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Run download_samanantar('{lang_pair}') first")
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        pairs.append((item["src"], item["tgt"], item["src_lang"], item["tgt_lang"]))
    return pairs


class TranslationDataset(Dataset):
    """
    PyTorch Dataset that tokenizes source/target pairs on the fly.
    """

    def __init__(self, pairs: List[Tuple], tokenizer: ArthSathiTokenizer, max_len: int = MAX_LEN):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text, src_lang, tgt_lang = self.pairs[idx]

        src_ids = self.tok.encode(src_text, lang=src_lang, max_length=self.max_len)
        tgt_ids = self.tok.encode(tgt_text, lang=tgt_lang, max_length=self.max_len)

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
        }


def collate_fn(batch, pad_id: int = 0):
    """Pad sequences in a batch to the same length."""
    src_max = max(x["src"].size(0) for x in batch)
    tgt_max = max(x["tgt"].size(0) for x in batch)

    src_padded = torch.full((len(batch), src_max), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_max), pad_id, dtype=torch.long)

    for i, item in enumerate(batch):
        src_padded[i, : item["src"].size(0)] = item["src"]
        tgt_padded[i, : item["tgt"].size(0)] = item["tgt"]

    return {"src": src_padded, "tgt": tgt_padded}


if __name__ == "__main__":
    # Run this to download corpus
    download_samanantar("hi", max_samples=100000)
    download_samanantar("mr", max_samples=50000)
    corpus_path = build_plain_text_corpus()
    print(f"\nNext: train tokenizer\n  python translation/tokenizer.py --corpus {corpus_path}")
