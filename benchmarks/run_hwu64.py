"""HWU64: zero-shot v0.3.0 + domain-adapted fine-tune. Official test untouched."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from onesystem.benchmarks import (
    TASK,
    evaluate_intent,
    load_hwu64,
    split_train_calib,
    to_onesystem_record,
    write_jsonl,
)
from onesystem.domain_adapt import domain_adapt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmarks" / "hwu64"
OUT = ROOT / "benchmarks" / "results"
ADAPTER_DIR = ROOT / "models" / "onesystem-hwu64"

BASE = "https://raw.githubusercontent.com/jianguoz/Few-Shot-Intent-Detection/main/Datasets/HWU64"


def _download() -> None:
    for split in ("train", "test"):
        dest = DATA / split
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("seq.in", "label"):
            path = dest / name
            if path.is_file() and path.stat().st_size > 0:
                continue
            url = f"{BASE}/{split}/{name}"
            print(f"download {url}", flush=True)
            urllib.request.urlretrieve(url, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    from onesystem import __version__
    from onesystem.model import OneSystem
    from onesystem.train import pick_device

    _download()
    data = load_hwu64(DATA / "train", DATA / "test")
    labels = data["labels"]
    train_rows, test_rows = data["train"], data["test"]
    assert len(labels) == 64, len(labels)

    device = args.device or pick_device()
    report = {
        "dataset": "HWU64",
        "source": "https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/HWU64",
        "task": TASK,
        "n_labels": len(labels),
        "label_wording": "raw intent strings from seq.in/label (frozen)",
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "protocol": "Every test example sees all 64 candidate labels. Official test never used for training or calibration.",
        "onesystem_version": __version__,
        "device": device,
    }
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.skip_zero_shot:
        print(f"loading base OneSystem on {device}", flush=True)
        base = OneSystem.load(device=device)
        print("HWU64 zero-shot / current checkpoint …", flush=True)
        t0 = time.time()
        zs = evaluate_intent(base, test_rows, labels, task=TASK)
        zs["seconds"] = round(time.time() - t0, 1)
        report["current_checkpoint"] = zs
        print(json.dumps(zs, indent=2), flush=True)

    if not args.skip_finetune:
        train_part, calib_part = split_train_calib(train_rows, calib_ratio=0.1, seed=42)
        train_jsonl = DATA / "train_onesystem.jsonl"
        calib_jsonl = DATA / "calib_onesystem.jsonl"
        write_jsonl(train_jsonl, [to_onesystem_record(r["text"], r["category"], labels) for r in train_part])
        write_jsonl(calib_jsonl, [to_onesystem_record(r["text"], r["category"], labels) for r in calib_part])
        report["finetune_split"] = {
            "train_rows": len(train_part),
            "calib_rows": len(calib_part),
            "calib_ratio": 0.1,
            "seed": 42,
        }
        from onesystem.data import read_jsonl

        model, manifest = domain_adapt(
            read_jsonl(train_jsonl),
            read_jsonl(calib_jsonl),
            ADAPTER_DIR,
            name="OneSystem-HWU64",
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_text_len=512,
            max_label_len=64,
            device=device,
        )
        print("HWU64 domain-adapted eval on official test …", flush=True)
        t0 = time.time()
        ft = evaluate_intent(model, test_rows, labels, task=TASK)
        ft["seconds"] = round(time.time() - t0, 1)
        ft["temperature"] = manifest["temperature"]
        ft["calib_nll"] = manifest["calib_nll"]
        ft["train_run"] = manifest["train_run"]
        report["domain_adapted"] = ft
        print(json.dumps(ft, indent=2), flush=True)

    path = OUT / "hwu64.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
