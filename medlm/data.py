"""Token packing for MedLM pretrain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """Memmap of uint16/uint32 token ids; returns (x, y) of length block_size."""

    def __init__(self, bin_path: Path, block_size: int):
        self.block_size = block_size
        path = Path(bin_path)
        meta_path = path.with_suffix(".meta.json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            dtype = np.uint16 if meta.get("dtype", "uint16") == "uint16" else np.uint32
        else:
            dtype = np.uint16
        self.data = np.memmap(path, dtype=dtype, mode="r")
        if len(self.data) <= block_size + 1:
            raise ValueError(f"{path} too short for block_size={block_size}")

    def __len__(self) -> int:
        # many random windows per epoch
        return max(1, (len(self.data) - self.block_size - 1) // self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = (idx * self.block_size) % (len(self.data) - self.block_size - 1)
        chunk = torch.from_numpy(self.data[i : i + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def iter_text_files(data_dir: Path) -> Iterator[str]:
    for path in sorted(data_dir.rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            yield text
    for path in sorted(data_dir.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, str):
                    yield row
                elif isinstance(row, dict):
                    for key in ("text", "input", "content", "document"):
                        if key in row and isinstance(row[key], str) and row[key].strip():
                            yield row[key]
                            break


def encode_corpus_to_bin(
    texts: List[str],
    tokenizer,
    out_bin: Path,
    *,
    dtype: str = "uint16",
) -> dict:
    """Encode texts to a flat token binary + meta."""
    ids: List[int] = []
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        eos = tokenizer.token_to_id("</s>")
    for text in texts:
        enc = tokenizer.encode(text).ids
        ids.extend(enc)
        if eos is not None:
            ids.append(eos)
    arr_dtype = np.uint16 if dtype == "uint16" else np.uint32
    if dtype == "uint16" and (max(ids) if ids else 0) >= 65535:
        arr_dtype = np.uint32
        dtype = "uint32"
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    mem = np.memmap(out_bin, dtype=arr_dtype, mode="w+", shape=(len(ids),))
    mem[:] = np.array(ids, dtype=arr_dtype)
    mem.flush()
    meta = {"tokens": len(ids), "dtype": dtype, "path": str(out_bin)}
    out_bin.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
