#!/usr/bin/env python3
"""Generate text from a MedLM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medlm.model import MedLM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/medlm-120m")
    parser.add_argument("--prompt", default="Patient presents with cough.")
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model_dir = Path(args.model)
    model = MedLM.from_pretrained(model_dir, map_location="cpu").to(device)
    model.eval()
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    ids = tok.encode(args.prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=args.tokens)
    text = tok.decode(out[0].tolist())
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
