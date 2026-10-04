"""Head-to-head comparison: OneSystem vs Laya vs OpenJev (Verdict).

Tracks
------
* **raw** — released / default checkpoints, no domain fine-tune
* **fine_tuned** — OneSystem domain checkpoints (Laya/OpenJev fine-tune N/A
  in public packages; reported as such)

Datasets: BANKING77, CLINC150 (in-scope), HWU64 official tests.

Usage
-----
# OneSystem (transformers<5 venv)
.venv/bin/python benchmarks/run_comparison.py --systems onesystem --track raw
.venv/bin/python benchmarks/run_comparison.py --systems onesystem --track fine_tuned

# Laya + OpenJev (compare venv)
.venv-compare/bin/python benchmarks/run_comparison.py --systems laya,openjev --track raw

# Merge + write docs JSON
.venv/bin/python benchmarks/run_comparison.py --merge
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

# Metrics helpers live in onesystem; for competitor-only venvs we vendor a copy path.
try:
    from onesystem.benchmarks import (
        expected_calibration_error,
        load_clinc150,
        load_hwu64,
        macro_f1,
        read_banking77_csv,
        banking77_labels,
    )
except ModuleNotFoundError:
    # Minimal fallbacks for .venv-compare (no onesystem installed)
    import csv
    import random
    from collections import defaultdict as _dd

    def read_banking77_csv(path):
        with Path(path).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def banking77_labels(rows):
        return sorted({r["category"] for r in rows})

    def load_hwu64(train_path, test_path):
        def read_pair(seq_path, label_path):
            texts = Path(seq_path).read_text(encoding="utf-8").splitlines()
            labs = Path(label_path).read_text(encoding="utf-8").splitlines()
            return [{"text": t.strip(), "category": l.strip()} for t, l in zip(texts, labs) if t.strip() and l.strip()]

        train = read_pair(Path(train_path) / "seq.in", Path(train_path) / "label")
        test = read_pair(Path(test_path) / "seq.in", Path(test_path) / "label")
        labels = sorted({r["category"] for r in train + test})
        return {"train": train, "test": test, "labels": labels}

    def load_clinc150(path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        def pack(pairs, oos=False):
            return [{"text": t, "category": "oos" if oos else lab} for t, lab in pairs]

        labels = sorted({lab for _, lab in raw["train"]})
        return {
            "labels": labels,
            "train": pack(raw["train"]),
            "val": pack(raw["val"]),
            "test": pack(raw["test"]),
            "oos_train": pack(raw["oos_train"], True),
            "oos_val": pack(raw["oos_val"], True),
            "oos_test": pack(raw["oos_test"], True),
        }

    def expected_calibration_error(confidences, hits, bins=10):
        if not confidences:
            return 0.0
        total = len(confidences)
        error = 0.0
        for b in range(bins):
            low, high = b / bins, (b + 1) / bins
            members = [i for i, c in enumerate(confidences) if (c > low or (b == 0 and c == low)) and c <= high]
            if not members:
                continue
            accuracy = sum(hits[i] for i in members) / len(members)
            confidence = sum(confidences[i] for i in members) / len(members)
            error += abs(accuracy - confidence) * len(members) / total
        return error

    def macro_f1(y_true, y_pred, labels):
        scores = []
        for label in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        return sum(scores) / len(scores) if scores else 0.0

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmarks"
OUT = ROOT / "benchmarks" / "results" / "comparison"
DOCS = ROOT / "docs" / "data"


def _latency_ms(adapter, texts: Sequence[str], labels: Sequence[str], repeats: int = 30) -> Dict:
    adapter.warm(texts, labels, n=min(3, len(texts)))
    times = []
    for i in range(repeats):
        t0 = time.perf_counter()
        adapter.classify_intent(texts[i % len(texts)], labels)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "n": repeats,
        "p50_ms": round(times[len(times) // 2], 2),
        "p95_ms": round(times[max(0, int(math.ceil(0.95 * len(times)) - 1))], 2),
        "mean_ms": round(sum(times) / len(times), 2),
    }


def evaluate_adapter(adapter, rows: List[Dict], labels: List[str], *, max_examples: int = 0) -> Dict:
    if max_examples and max_examples < len(rows):
        rows = rows[:max_examples]
    y_true, y_pred = [], []
    confidences, hits, nlls = [], [], []
    errors = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        ans = adapter.classify_intent(row["text"], labels)
        gold = row["category"]
        y_true.append(gold)
        if ans.error:
            errors += 1
        pred = ans.label
        y_pred.append(pred if pred is not None else "__none__")
        if pred is not None and ans.probabilities:
            confidences.append(ans.confidence)
            hit = int(str(pred) == gold)
            hits.append(hit)
            p = max(float(ans.probabilities.get(gold, 0.0)), 1e-12)
            nlls.append(-math.log(p))
        if (i + 1) % 100 == 0:
            print(f"  {adapter.name}: {i+1}/{len(rows)}", flush=True)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    result = {
        "system": adapter.name,
        "track": adapter.track,
        "n": len(rows),
        "n_labels": len(labels),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1(y_true, y_pred, labels), 4),
        "ece": round(expected_calibration_error(confidences, hits), 4) if confidences else None,
        "nll": round(sum(nlls) / len(nlls), 4) if nlls else None,
        "error_rate": round(errors / len(rows), 4) if rows else 0.0,
        "seconds": round(time.time() - t0, 1),
    }
    try:
        result["latency"] = _latency_ms(adapter, [r["text"] for r in rows], labels)
    except Exception as exc:  # noqa: BLE001
        result["latency_error"] = str(exc)
    return result


def load_datasets():
    banking_train = read_banking77_csv(DATA / "banking77" / "train.csv")
    banking_test = read_banking77_csv(DATA / "banking77" / "test.csv")
    banking_labels = banking77_labels(banking_train + banking_test)
    banking_rows = [{"text": r["text"], "category": r["category"]} for r in banking_test]

    clinc = load_clinc150(DATA / "clinc150" / "data_full.json")
    hwu = load_hwu64(DATA / "hwu64" / "train", DATA / "hwu64" / "test")
    return {
        "banking77": {"rows": banking_rows, "labels": banking_labels},
        "clinc150": {"rows": clinc["test"], "labels": clinc["labels"]},
        "hwu64": {"rows": hwu["test"], "labels": hwu["labels"]},
    }


def make_adapters(systems: Sequence[str], track: str, device: str | None):
    # Local import so Laya/OpenJev envs need not install onesystem.
    sys_path_root = str(ROOT)
    if sys_path_root not in __import__("sys").path:
        __import__("sys").path.insert(0, sys_path_root)
    from benchmarks.compare_adapters import LayaAdapter, OneSystemAdapter, OpenJevAdapter

    adapters = []
    for sys_name in systems:
        if sys_name == "onesystem" and track == "raw":
            adapters.append(OneSystemAdapter(source=None, device=device, track="raw"))
        elif sys_name == "onesystem" and track == "fine_tuned":
            for key, path in (
                ("banking77", ROOT / "models" / "onesystem-banking77"),
                ("clinc150", ROOT / "models" / "onesystem-clinc150"),
                ("hwu64", ROOT / "models" / "onesystem-hwu64"),
            ):
                if (path / "model.safetensors").is_file():
                    a = OneSystemAdapter(source=str(path), device=device, track="fine_tuned")
                    a.name = f"onesystem-{key}"
                    a.dataset_key = key  # type: ignore[attr-defined]
                    adapters.append(a)
        elif sys_name == "laya" and track == "raw":
            adapters.append(LayaAdapter(track="raw"))
        elif sys_name == "openjev" and track == "raw":
            adapters.append(OpenJevAdapter(track="raw"))
        elif sys_name in {"laya", "openjev"} and track == "fine_tuned":
            print(f"{sys_name}: no public domain fine-tune path — skip FT track", flush=True)
    return adapters


def merge_partials() -> Dict:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "title": "OneSystem vs Laya vs OpenJev",
        "protocol": (
            "Official test splits. Every example sees the full candidate label set. "
            "Raw = released/default checkpoints. Fine-tuned = OneSystem domain checkpoints "
            "trained only on that dataset's train split; Laya/OpenJev public packages have "
            "no equivalent documented domain_adapt path in this harness."
        ),
        "systems": {
            "onesystem": "OneSystem v0.3.0 (Alibaba-NLP/gte-base-en-v1.5 init)",
            "laya": "pip install laya → convaiinnovations/laya (english)",
            "openjev": "Verdict-open-jev (heman10x/rlcd-modernbert-151m) — open OpenJev",
        },
        "tracks": {},
        "notes": [
            "PyPI package `openjev==0.0.1` is an empty stub; this board uses Verdict-open-jev.",
            "TypeSafe hosted Jev is not evaluated (API key / waitlist).",
            "Holdout fast-decisions scores are a different evaluation and are not mixed in here.",
        ],
    }
    for path in sorted(OUT.glob("partial_*.json")):
        chunk = json.loads(path.read_text(encoding="utf-8"))
        track = chunk["track"]
        report["tracks"].setdefault(track, {})
        for dataset, rows in chunk["results"].items():
            report["tracks"][track].setdefault(dataset, [])
            # replace same system name if re-run
            existing = {r["system"]: i for i, r in enumerate(report["tracks"][track][dataset])}
            for row in rows:
                if row["system"] in existing:
                    report["tracks"][track][dataset][existing[row["system"]]] = row
                else:
                    report["tracks"][track][dataset].append(row)
    out_path = OUT / "comparison.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} and docs/data/comparison.json", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="onesystem", help="comma list: onesystem,laya,openjev")
    parser.add_argument("--track", choices=["raw", "fine_tuned"], default="raw")
    parser.add_argument("--datasets", default="banking77,clinc150,hwu64")
    parser.add_argument("--max-examples", type=int, default=0, help="cap per dataset (0=all)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    if args.merge:
        merge_partials()
        return

    # Allow importing benchmarks.compare_adapters when run as a script
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    want = {d.strip() for d in args.datasets.split(",") if d.strip()}
    datasets = {k: v for k, v in load_datasets().items() if k in want}
    adapters = make_adapters(systems, args.track, args.device)
    if not adapters:
        raise SystemExit("no adapters constructed")

    results: Dict[str, List] = defaultdict(list)
    for adapter in adapters:
        for ds_name, pack in datasets.items():
            # fine-tuned onesystem-* only eval its own dataset
            ds_key = getattr(adapter, "dataset_key", None)
            if ds_key and ds_key != ds_name:
                continue
            print(f"== {adapter.name} / {args.track} / {ds_name} ({len(pack['rows'])} ex, {len(pack['labels'])} labels)", flush=True)
            metrics = evaluate_adapter(adapter, pack["rows"], pack["labels"], max_examples=args.max_examples)
            metrics["dataset"] = ds_name
            results[ds_name].append(metrics)
            print(json.dumps(metrics, indent=2), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.track}_{'_'.join(systems)}"
    path = OUT / f"partial_{tag}.json"
    # Merge with existing partial so a single-dataset re-run does not wipe others.
    merged = {"track": args.track, "systems": systems, "results": {}}
    if path.is_file():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
            merged["results"] = dict(prev.get("results") or {})
        except json.JSONDecodeError:
            pass
    for ds_name, rows in results.items():
        by_system = {r["system"]: r for r in merged["results"].get(ds_name, [])}
        for row in rows:
            by_system[row["system"]] = row
        merged["results"][ds_name] = list(by_system.values())
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
