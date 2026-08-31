"""
BLEU evaluation for our trained translation model.
Compares our model vs a naive baseline (copy source = 0 BLEU floor).

Member C owns this file.

Usage:
    python translation/evaluate_bleu.py --lang hi
"""

import argparse
import json
from pathlib import Path

from sacrebleu.metrics import BLEU
from translation.inference import Translator
from translation.dataset import load_pairs

bleu_metric = BLEU(effective_order=True)


def evaluate(lang: str = "hi", n_samples: int = 1000):
    print(f"Loading test pairs for {lang}-en...")
    all_pairs = load_pairs(lang)

    # Use last n_samples as test (were not in training)
    test_pairs = all_pairs[-n_samples:]
    print(f"Evaluating on {len(test_pairs)} sentence pairs")

    # Filter to just hi→en direction for cleaner eval
    test_pairs = [(s, t, sl, tl) for s, t, sl, tl in test_pairs if sl == lang and tl == "en"]
    print(f"  {len(test_pairs)} {lang}→en pairs")

    if not test_pairs:
        print("No pairs found. Check dataset.")
        return

    translator = Translator()

    hypotheses = []
    references = []

    for i, (src, ref, src_lang, tgt_lang) in enumerate(test_pairs[:500]):  # cap at 500 for speed
        try:
            hyp = translator.translate(src, src_lang, tgt_lang, beam=True)
        except Exception:
            hyp = ""
        hypotheses.append(hyp)
        references.append(ref)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{min(500, len(test_pairs))} done...")

    # Compute corpus BLEU
    score = bleu_metric.corpus_score(hypotheses, [references])
    print(f"\n── BLEU Results ({lang}→en) ──────────────────────")
    print(f"  Our model BLEU : {score.score:.2f}")

    # Naive baseline: copy source as translation (should score ~0)
    baseline_score = bleu_metric.corpus_score(
        [src for src, _, _, _ in test_pairs[:len(hypotheses)]],
        [references]
    )
    print(f"  Copy baseline  : {baseline_score.score:.2f}")
    print(f"  Delta          : +{score.score - baseline_score.score:.2f}")

    # Show 3 examples
    print("\n── Sample Translations ────────────────────────────")
    for src, ref, sl, tl in test_pairs[:3]:
        hyp = translator.translate(src, sl, tl)
        print(f"  Source ({sl}): {src[:80]}")
        print(f"  Reference   : {ref[:80]}")
        print(f"  Our model   : {hyp[:80]}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="hi", choices=["hi", "mr"])
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()
    evaluate(args.lang, args.n_samples)
