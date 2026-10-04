"""GoEmotions: multi-label emotion tagging — zero-shot + domain-adapted."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, List

from onesystem.benchmarks import evaluate_multilabel, write_jsonl
from onesystem.domain_adapt import domain_adapt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmarks" / "goemotions"
OUT = ROOT / "benchmarks" / "results"
ADAPTER_DIR = ROOT / "models" / "onesystem-goemotions"

SPLIT_BASE = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data"
TASK = "emotion"

# Official 28-way taxonomy (index order matches emotion ids in the TSVs).
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read().decode("utf-8")


def _parse_split(text: str, skip_neutral_only: bool = True) -> List[Dict]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        utterance, id_field = parts[0].strip(), parts[1].strip()
        if not utterance:
            continue
        ids = []
        for token in id_field.replace(" ", "").split(","):
            if token.isdigit():
                ids.append(int(token))
        labs = [EMOTIONS[i] for i in ids if 0 <= i < len(EMOTIONS)]
        if not labs:
            continue
        if skip_neutral_only and labs == ["neutral"]:
            continue
        rows.append({"text": utterance, "labels": labs})
    return rows


def _download_splits() -> Dict[str, List[Dict]]:
    DATA.mkdir(parents=True, exist_ok=True)
    out: Dict[str, List[Dict]] = {}
    for split in ("train", "dev", "test"):
        path = DATA / f"{split}.tsv"
        if not path.is_file():
            url = f"{SPLIT_BASE}/{split}.tsv"
            print(f"download {url}", flush=True)
            path.write_text(_fetch(url), encoding="utf-8")
        rows = _parse_split(path.read_text(encoding="utf-8"))
        out[split] = rows
        print(f"{split}: {len(rows)} rows (neutral-only skipped)", flush=True)
    return out


def _to_record(text: str, labels: List[str]) -> Dict:
    # Candidate set excludes exclusive-neutral so the head focuses on emotions.
    candidates = [e for e in EMOTIONS if e != "neutral"]
    gold = [lab for lab in labels if lab != "neutral"] or labels
    return {
        "input": text,
        "output": {
            "classifications": [
                {
                    "task": TASK,
                    "labels": candidates,
                    "true_label": gold,
                    "multi_label": True,
                }
            ]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-train", type=int, default=0, help="optional cap for faster smoke runs")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    from onesystem import __version__
    from onesystem.data import read_jsonl
    from onesystem.model import OneSystem
    from onesystem.train import pick_device

    splits = _download_splits()
    train_rows = splits["train"]
    calib_src = splits["dev"]
    test_rows = splits["test"]
    if args.max_train and args.max_train < len(train_rows):
        train_rows = train_rows[: args.max_train]

    label_set = [e for e in EMOTIONS if e != "neutral"]
    device = args.device or pick_device()
    report = {
        "dataset": "GoEmotions",
        "source": "https://github.com/google-research/google-research/tree/master/goemotions",
        "task": TASK,
        "n_labels": len(label_set),
        "label_wording": "27 GoEmotions labels (neutral-only rows skipped; frozen names)",
        "train_rows": len(train_rows),
        "calib_rows": min(len(calib_src), 2000),
        "test_rows": len(test_rows),
        "protocol": (
            "Multi-label: every example sees all 27 emotion candidates. "
            "Official test untouched. Dev (capped) used only for temperature calibration."
        ),
        "onesystem_version": __version__,
        "device": device,
        "threshold": args.threshold,
    }
    OUT.mkdir(parents=True, exist_ok=True)

    def strip_neutral(rows: List[Dict]) -> List[Dict]:
        cleaned = []
        for row in rows:
            labs = [lab for lab in row["labels"] if lab != "neutral"]
            if labs:
                cleaned.append({"text": row["text"], "labels": labs})
        return cleaned

    train_rows = strip_neutral(train_rows)
    calib_src = strip_neutral(calib_src)
    test_rows = strip_neutral(test_rows)
    report["train_rows"] = len(train_rows)
    report["test_rows"] = len(test_rows)

    if not args.skip_zero_shot:
        print(f"loading base OneSystem on {device}", flush=True)
        base = OneSystem.load(device=device)
        print("GoEmotions zero-shot / current checkpoint …", flush=True)
        t0 = time.time()
        zs = evaluate_multilabel(base, test_rows, label_set, task=TASK, threshold=args.threshold)
        zs["seconds"] = round(time.time() - t0, 1)
        report["current_checkpoint"] = zs
        print(json.dumps(zs, indent=2), flush=True)

    if not args.skip_finetune:
        train_jsonl = DATA / "train_onesystem.jsonl"
        calib_jsonl = DATA / "calib_onesystem.jsonl"
        write_jsonl(train_jsonl, [_to_record(r["text"], r["labels"]) for r in train_rows])
        calib_rows = calib_src[:2000]
        write_jsonl(calib_jsonl, [_to_record(r["text"], r["labels"]) for r in calib_rows])
        report["finetune_split"] = {
            "train_rows": len(train_rows),
            "calib_rows": len(calib_rows),
            "calib_source": "official_dev",
        }
        report["calib_rows"] = len(calib_rows)
        model, manifest = domain_adapt(
            read_jsonl(train_jsonl),
            read_jsonl(calib_jsonl),
            ADAPTER_DIR,
            name="OneSystem-GoEmotions",
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_text_len=256,
            max_label_len=64,
            device=device,
        )
        print("GoEmotions domain-adapted eval on official test …", flush=True)
        t0 = time.time()
        ft = evaluate_multilabel(model, test_rows, label_set, task=TASK, threshold=args.threshold)
        ft["seconds"] = round(time.time() - t0, 1)
        ft["temperature"] = manifest["temperature"]
        ft["calib_nll"] = manifest["calib_nll"]
        ft["train_run"] = manifest["train_run"]
        report["domain_adapted"] = ft
        print(json.dumps(ft, indent=2), flush=True)

    path = OUT / "goemotions.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
