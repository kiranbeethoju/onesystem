"""Public-benchmark helpers: metrics, latency, and intent/OOS evaluation."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TASK = "intent"


def read_banking77_csv(path: Path | str) -> List[Dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def banking77_labels(rows: Sequence[Dict[str, str]]) -> List[str]:
    return sorted({row["category"] for row in rows})


def to_onesystem_record(text: str, label: str, labels: Sequence[str], task: str = TASK) -> Dict:
    return {
        "input": text,
        "output": {
            "classifications": [
                {
                    "task": task,
                    "labels": list(labels),
                    "true_label": [label],
                    "multi_label": False,
                }
            ]
        },
    }


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_train_calib(
    rows: Sequence[Dict[str, str]],
    calib_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Stratified-ish split: shuffle within each label, then cut calib."""
    by_label: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["category"]].append(row)
    rng = random.Random(seed)
    train, calib = [], []
    for label, items in sorted(by_label.items()):
        items = list(items)
        rng.shuffle(items)
        n_calib = max(1, int(round(len(items) * calib_ratio))) if len(items) > 1 else 0
        calib.extend(items[:n_calib])
        train.extend(items[n_calib:])
    rng.shuffle(train)
    rng.shuffle(calib)
    return train, calib


def expected_calibration_error(confidences: Sequence[float], hits: Sequence[int], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    total = len(confidences)
    error = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = [
            i
            for i, c in enumerate(confidences)
            if (c > low or (b == 0 and c == low)) and c <= high
        ]
        if not members:
            continue
        accuracy = sum(hits[i] for i in members) / len(members)
        confidence = sum(confidences[i] for i in members) / len(members)
        error += abs(accuracy - confidence) * len(members) / total
    return error


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return sum(scores) / len(scores) if scores else 0.0


def nll_from_probs(gold: str, probabilities: Dict[str, float]) -> float:
    p = max(float(probabilities.get(gold, 0.0)), 1e-12)
    return -math.log(p)


def warm_latency_ms(model, texts: Sequence[str], labels: Sequence[str], task: str = TASK, repeats: int = 50) -> Dict:
    """Warm the model, then time single-example classify calls."""
    tasks = {task: list(labels)}
    # warmup
    for text in texts[: min(5, len(texts))]:
        model.classify(text, tasks)
    sample = [texts[i % len(texts)] for i in range(repeats)]
    times = []
    for text in sample:
        t0 = time.perf_counter()
        model.classify(text, tasks)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[max(0, int(math.ceil(0.95 * len(times)) - 1))]
    return {
        "n": repeats,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "mean_ms": round(sum(times) / len(times), 2),
    }


def evaluate_intent(
    model,
    rows: Sequence[Dict[str, str]],
    labels: Sequence[str],
    task: str = TASK,
    text_key: str = "text",
    label_key: str = "category",
    min_confidence: Optional[float] = None,
    measure_latency: bool = True,
) -> Dict:
    """Score single-label intent classification over a fixed candidate set."""
    tasks = {task: {"labels": list(labels)}}
    y_true: List[str] = []
    y_pred: List[str] = []
    confidences: List[float] = []
    hits: List[int] = []
    nlls: List[float] = []
    accepted_hits = 0
    accepted = 0
    abstained = 0

    batch_size = 32
    texts = [row[text_key] for row in rows]
    golds = [row[label_key] for row in rows]
    for start in range(0, len(rows), batch_size):
        chunk_texts = texts[start:start + batch_size]
        chunk_golds = golds[start:start + batch_size]
        answers = model.classify_batch(chunk_texts, tasks, min_confidence=min_confidence)
        for gold, answer in zip(chunk_golds, answers):
            answer = answer[task]
            pred = answer.get("label")
            conf = float(answer.get("confidence") or 0.0)
            probs = answer.get("probabilities") or {}
            y_true.append(gold)
            if pred is None or answer.get("abstain"):
                abstained += 1
                y_pred.append("__abstain__")
            else:
                accepted += 1
                y_pred.append(str(pred))
                hit = int(str(pred) == gold)
                accepted_hits += hit
                hits.append(hit)
                confidences.append(conf)
                nlls.append(nll_from_probs(gold, probs))

    # For accuracy/F1 on forced predictions (no abstain), re-score without floor.
    if min_confidence is not None:
        forced = evaluate_intent(
            model, rows, labels, task=task, text_key=text_key, label_key=label_key, min_confidence=None, measure_latency=False
        )
        forced_acc = forced["accuracy"]
        forced_f1 = forced["macro_f1"]
        forced_ece = forced["ece"]
        forced_nll = forced["nll"]
    else:
        forced_acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
        forced_f1 = macro_f1(y_true, y_pred, labels)
        forced_ece = expected_calibration_error(confidences, hits)
        forced_nll = sum(nlls) / len(nlls) if nlls else float("nan")

    result = {
        "n": len(rows),
        "n_labels": len(labels),
        "accuracy": round(forced_acc, 4),
        "macro_f1": round(forced_f1, 4),
        "ece": round(forced_ece, 4),
        "nll": round(forced_nll, 4) if forced_nll == forced_nll else None,
        "coverage": round(accepted / len(rows), 4) if rows else 0.0,
        "accepted_accuracy": round(accepted_hits / accepted, 4) if accepted else None,
        "abstain_rate": round(abstained / len(rows), 4) if rows else 0.0,
        "min_confidence": min_confidence,
    }
    if measure_latency and rows:
        result["latency"] = warm_latency_ms(model, [r[text_key] for r in rows], labels, task=task)
    return result


def evaluate_oos(
    model,
    in_scope: Sequence[Dict[str, str]],
    out_of_scope: Sequence[Dict[str, str]],
    labels: Sequence[str],
    min_confidence: float,
    task: str = TASK,
    text_key: str = "text",
    label_key: str = "category",
) -> Dict:
    """Abstention policy on in-scope vs OOS. Threshold must be chosen on validation."""
    tasks = {task: {"labels": list(labels)}}
    batch_size = 32
    oos_rejected = 0
    oos_texts = [row[text_key] for row in out_of_scope]
    for start in range(0, len(oos_texts), batch_size):
        for answer in model.classify_batch(oos_texts[start:start + batch_size], tasks, min_confidence=min_confidence):
            answer = answer[task]
            if answer.get("abstain") or answer.get("label") is None:
                oos_rejected += 1
    in_rejected = 0
    in_accepted_correct = 0
    in_accepted = 0
    in_texts = [row[text_key] for row in in_scope]
    in_golds = [row[label_key] for row in in_scope]
    for start in range(0, len(in_texts), batch_size):
        chunk_answers = model.classify_batch(in_texts[start:start + batch_size], tasks, min_confidence=min_confidence)
        for gold, answer in zip(in_golds[start:start + batch_size], chunk_answers):
            answer = answer[task]
            if answer.get("abstain") or answer.get("label") is None:
                in_rejected += 1
            else:
                in_accepted += 1
                if str(answer["label"]) == gold:
                    in_accepted_correct += 1
    return {
        "min_confidence": min_confidence,
        "oos_n": len(out_of_scope),
        "oos_recall": round(oos_rejected / len(out_of_scope), 4) if out_of_scope else None,
        "in_scope_n": len(in_scope),
        "in_scope_rejection_rate": round(in_rejected / len(in_scope), 4) if in_scope else None,
        "coverage": round(in_accepted / len(in_scope), 4) if in_scope else None,
        "accepted_accuracy": round(in_accepted_correct / in_accepted, 4) if in_accepted else None,
    }


def choose_threshold(
    model,
    in_scope_val: Sequence[Dict[str, str]],
    oos_val: Sequence[Dict[str, str]],
    labels: Sequence[str],
    grid: Sequence[float] = tuple(round(x * 0.05, 2) for x in range(2, 19)),
    task: str = TASK,
    max_in_scope_rejection: float = 0.25,
) -> Tuple[float, Dict]:
    """Pick confidence floor on validation (never on test).

    Prefer thresholds with in-scope rejection ≤ ``max_in_scope_rejection``,
    maximising OOS recall among those. If none qualify, maximise
    ``oos_recall - in_scope_rejection_rate`` (Youden-style balance).
    """
    scored = []
    for t in grid:
        metrics = evaluate_oos(model, in_scope_val, oos_val, labels, min_confidence=t, task=task)
        scored.append(metrics)

    eligible = [m for m in scored if (m["in_scope_rejection_rate"] or 1.0) <= max_in_scope_rejection]
    pool = eligible or scored

    def key(m):
        oos = m["oos_recall"] or 0.0
        rej = m["in_scope_rejection_rate"] or 1.0
        acc = m["accepted_accuracy"] or 0.0
        if eligible:
            return (oos, acc, -(rej))
        return (oos - rej, acc, oos)

    best = max(pool, key=key)
    return float(best["min_confidence"]), {
        "grid": scored,
        "chosen": best["min_confidence"],
        "max_in_scope_rejection": max_in_scope_rejection,
        "eligible_count": len(eligible),
    }


def load_clinc150(path: Path) -> Dict[str, List[Dict[str, str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    def pack(pairs: List, oos: bool = False) -> List[Dict[str, str]]:
        rows = []
        for text, label in pairs:
            rows.append({"text": text, "category": "oos" if oos else label})
        return rows

    labels = sorted({label for _, label in raw["train"]})
    return {
        "labels": labels,
        "train": pack(raw["train"]),
        "val": pack(raw["val"]),
        "test": pack(raw["test"]),
        "oos_train": pack(raw["oos_train"], oos=True),
        "oos_val": pack(raw["oos_val"], oos=True),
        "oos_test": pack(raw["oos_test"], oos=True),
    }
