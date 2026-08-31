"""
Train and use a shared BPE tokenizer for Hindi + Marathi + English.
We train this ourselves on our corpus — not using a pretrained tokenizer.

Member C owns this file.
"""

import os
import json
from pathlib import Path
from typing import List

TOKENIZER_DIR = Path("translation/tokenizer_model")
VOCAB_SIZE = 16000   # small enough for CPU, covers Hindi+Marathi+English


def train_tokenizer(corpus_file: str, output_dir: str = str(TOKENIZER_DIR)) -> None:
    """
    Train a BPE tokenizer on our parallel corpus text.
    corpus_file: plain text file, one sentence per line (mix of all languages).
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[HI]", "[MR]", "[EN]"],
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train(files=[corpus_file], trainer=trainer)

    # Add post-processor to auto-add BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]")),
                        ("[EOS]", tokenizer.token_to_id("[EOS]"))],
    )
    tokenizer.decoder = decoders.BPEDecoder()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(Path(output_dir) / "tokenizer.json"))

    # Save vocab mapping for easy lookup
    vocab = tokenizer.get_vocab()
    with open(Path(output_dir) / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"✓ Tokenizer trained. Vocab size: {len(vocab)}")
    print(f"  Saved to: {output_dir}")


class ArthSathiTokenizer:
    """Wrapper around our trained BPE tokenizer."""

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    LANG_TOKENS = {"hi": "[HI]", "mr": "[MR]", "en": "[EN]"}

    def __init__(self, model_dir: str = str(TOKENIZER_DIR)):
        from tokenizers import Tokenizer
        path = Path(model_dir) / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {path}. Run: python translation/tokenizer.py"
            )
        self._tok = Tokenizer.from_file(str(path))

    def encode(self, text: str, lang: str | None = None, max_length: int = 256) -> List[int]:
        """Encode text to token IDs, optionally prepending language tag."""
        if lang and lang in self.LANG_TOKENS:
            text = f"{self.LANG_TOKENS[lang]} {text}"
        enc = self._tok.encode(text)
        ids = enc.ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        special = {self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID}
        if skip_special:
            ids = [i for i in ids if i not in special]
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        return self._tok.token_to_id(token)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Plain text file with all sentences")
    parser.add_argument("--output_dir", default=str(TOKENIZER_DIR))
    args = parser.parse_args()
    train_tokenizer(args.corpus, args.output_dir)
