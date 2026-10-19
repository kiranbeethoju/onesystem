#!/usr/bin/env python3
"""Train a BPE tokenizer on the MedLM corpus (tokenizers via transformers stack)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/medlm/raw")
    parser.add_argument("--out", default="data/medlm/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()

    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing
    from tokenizers.trainers import BpeTrainer

    data = Path(args.data)
    files = sorted(str(p) for p in data.glob("*.txt") if p.stat().st_size > 0)
    if not files:
        raise SystemExit(f"No .txt files in {data}; run: python -m medlm.prepare_corpus")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>", "<eos>", "<unk>", "<bos>"],
        show_progress=True,
    )
    tokenizer.train(files, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    print(f"Saved tokenizer vocab={tokenizer.get_vocab_size()} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
