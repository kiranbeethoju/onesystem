#!/usr/bin/env python3
"""Encode train/val text into packed token binaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medlm.data import encode_corpus_to_bin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/medlm/raw")
    parser.add_argument("--tokenizer", default="data/medlm/tokenizer.json")
    parser.add_argument("--out-dir", default="data/medlm")
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    data = Path(args.data)
    out_dir = Path(args.out_dir)
    for split in ("train", "val"):
        path = data / f"{split}.txt"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        # split on blank lines (documents)
        docs = [d.strip() for d in path.read_text(encoding="utf-8").split("\n\n") if d.strip()]
        meta = encode_corpus_to_bin(docs, tok, out_dir / f"{split}.bin")
        print(split, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
