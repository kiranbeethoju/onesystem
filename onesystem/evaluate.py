"""Score a saved OneSystem adapter on the local development holdout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from onesystem.data import EVAL_PATH, read_jsonl
from onesystem.model import ADAPTER_DIR, OneSystem


def tasks_from_record(record: Dict) -> Dict:
    tasks = {}
    for head in record["output"]["classifications"]:
        spec = {"labels": list(head["labels"])}
        if head.get("multi_label"):
            spec["multi_label"] = True
            spec["cls_threshold"] = 0.5
        tasks[head["task"]] = spec
    return tasks


def predicted_labels(answer) -> List[str]:
    if isinstance(answer, dict) and "label" in answer:
        label = answer["label"]
        if isinstance(label, list):
            return [str(item) for item in label]
        return [str(label)]
    if isinstance(answer, list):
        labels = []
        for item in answer:
            if isinstance(item, dict) and "label" in item:
                labels.append(str(item["label"]))
            else:
                labels.append(str(item))
        return labels
    if isinstance(answer, str):
        return [answer]
    return []


def gold_labels(head: Dict) -> List[str]:
    truth = head["true_label"]
    if isinstance(truth, str):
        return [truth]
    return [str(item) for item in truth]


def score_records(model: OneSystem, records: Iterable[Dict]) -> Dict[str, float]:
    heads = 0
    correct = 0
    by_task: Dict[str, List[int]] = {}
    for record in records:
        prediction = model.classify(record["input"], tasks_from_record(record))
        for head in record["output"]["classifications"]:
            task = head["task"]
            got = set(predicted_labels(prediction.get(task)))
            expected = set(gold_labels(head))
            hit = int(got == expected)
            heads += 1
            correct += hit
            by_task.setdefault(task, [0, 0])
            by_task[task][0] += hit
            by_task[task][1] += 1
    per_task = {
        task: {"correct": hits, "heads": total, "accuracy": hits / total}
        for task, (hits, total) in sorted(by_task.items())
    }
    return {
        "heads": heads,
        "correct": correct,
        "exact_match": (correct / heads) if heads else 0.0,
        "per_task": per_task,
    }


def main() -> None:
    records = read_jsonl(EVAL_PATH)
    model = OneSystem.load(ADAPTER_DIR)
    metrics = score_records(model, records)
    summary = {
        "name": model.name,
        "eval_rows": len(records),
        "exact_match": metrics["exact_match"],
        "heads": metrics["heads"],
        "correct": metrics["correct"],
    }
    print(json.dumps(summary, indent=2))
    destination = ADAPTER_DIR / "eval.json"
    details = dict(summary)
    details["per_task"] = metrics["per_task"]
    details["note"] = (
        "Scored on a holdout cut from the public development split. "
        "This is not the held-out fast-decisions test benchmark."
    )
    destination.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
