"""Score OneSystem on the local holdout and fit its confidence temperature."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from onesystem.data import CALIB_PATH, EVAL_PATH, read_jsonl
from onesystem.hub import LOCAL_MODEL_DIR


def tasks_from_record(record: Dict) -> Dict:
    tasks = {}
    for head in record["output"]["classifications"]:
        tasks[head["task"]] = {
            "labels": list(head["labels"]),
            "multi_label": bool(head.get("multi_label")),
            "threshold": 0.5,
        }
    return tasks


def gold_labels(head: Dict) -> List[str]:
    truth = head["true_label"]
    if isinstance(truth, str):
        return [truth]
    return [str(item) for item in truth]


def predicted_labels(answer: Dict) -> List[str]:
    if "labels" in answer:
        return [str(item) for item in answer["labels"]]
    label = answer.get("label")
    return [str(label)] if label is not None else []


def expected_calibration_error(confidences: List[float], hits: List[int], bins: int = 10) -> float:
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


def score_records(model, records: Iterable[Dict]) -> Dict:
    heads = correct = 0
    by_task: Dict[str, List[int]] = {}
    confidences: List[float] = []
    hits: List[int] = []
    for record in records:
        prediction = model.classify(record["input"], tasks_from_record(record))
        for head in record["output"]["classifications"]:
            task = head["task"]
            answer = prediction[task]
            got = set(predicted_labels(answer))
            expected = set(gold_labels(head))
            hit = int(got == expected)
            heads += 1
            correct += hit
            by_task.setdefault(task, [0, 0])
            by_task[task][0] += hit
            by_task[task][1] += 1
            if "confidence" in answer:
                confidences.append(float(answer["confidence"]))
                hits.append(hit)
    per_task = {
        task: {"correct": h, "heads": n, "accuracy": round(h / n, 4)}
        for task, (h, n) in sorted(by_task.items())
    }
    return {
        "heads": heads,
        "correct": correct,
        "exact_match": round(correct / heads, 4) if heads else 0.0,
        "ece": round(expected_calibration_error(confidences, hits), 4),
        "per_task": per_task,
    }


def fit_temperature(model, records: List[Dict], grid: Tuple[float, float, int] = (0.2, 10.0, 99)) -> Tuple[float, float]:
    """Choose the temperature that minimises NLL of the gold label on single-label heads.

    The model's raw logits are collected once with temperature 1, then rescaled
    on the grid, so the fit is cheap. Multi-label heads are excluded.
    """
    import torch

    from onesystem.modeling import tokenize

    original = float(model.network.config.temperature)
    model.network.config.temperature = 1.0
    model.clear_cache()
    logit_rows: List[torch.Tensor] = []
    targets: List[int] = []
    with torch.no_grad():
        for record in records:
            for head in record["output"]["classifications"]:
                if head.get("multi_label") or len(gold_labels(head)) != 1:
                    continue
                task = head["task"]
                labels = list(head["labels"])
                batch = tokenize(model.tokenizer, [model.network.text_prompt(task, record["input"])], model.network.config.max_text_len, model.device)
                text_embedding = model.network.encode_text(**batch)
                label_embeddings = model._label_embeddings(task, labels, {})
                logit_rows.append(model.network.scores(text_embedding, label_embeddings)[0].cpu())
                targets.append(labels.index(gold_labels(head)[0]))
    model.network.config.temperature = original
    model.clear_cache()
    if not logit_rows:
        return 1.0, float("nan")
    best_t, best_nll = 1.0, math.inf
    low, high, steps = grid
    for i in range(steps):
        t = low + (high - low) * i / (steps - 1)
        nll = 0.0
        for logits, target in zip(logit_rows, targets):
            nll -= float(torch.log_softmax(logits / t, dim=-1)[target])
        nll /= len(logit_rows)
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t, best_nll


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate OneSystem on the local holdout")
    parser.add_argument("--source", default=None, help="model directory, github:owner/repo@tag, or default")
    parser.add_argument("--eval", default=str(EVAL_PATH))
    parser.add_argument("--fit-temperature", action="store_true", help="refit the temperature on the calibration split and save it")
    args = parser.parse_args(argv)

    from onesystem.model import OneSystem

    model = OneSystem.load(args.source)
    if args.fit_temperature:
        temperature, nll = fit_temperature(model, read_jsonl(CALIB_PATH))
        model.network.config.temperature = temperature
        model.clear_cache()
        print(f"temperature {temperature:.3f} (calibration NLL {nll:.4f})")
    records = read_jsonl(Path(args.eval))
    metrics = score_records(model, records)
    summary = {k: metrics[k] for k in ("heads", "correct", "exact_match", "ece")}
    summary["name"] = model.name
    summary["eval_rows"] = len(records)
    print(json.dumps(summary, indent=2))

    destination = LOCAL_MODEL_DIR / "eval.json"
    if destination.parent.is_dir():
        details = {**metrics, "eval_rows": len(records), "note": (
            "Scored on a holdout cut from the public development split. "
            "This is not the held-out fast-decisions test benchmark."
        )}
        destination.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {destination}")
        if args.fit_temperature:
            model.network.config.save(LOCAL_MODEL_DIR)


if __name__ == "__main__":
    main()
