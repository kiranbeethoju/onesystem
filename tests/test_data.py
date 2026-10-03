import json
from pathlib import Path

import pytest

from onesystem.data import RecordError, read_jsonl, split_by_domain, validate_record, write_jsonl


def _row(text, task="intent", label="yes", labels=None):
    return {
        "input": text,
        "output": {
            "classifications": [
                {
                    "task": task,
                    "labels": labels or ["yes", "no"],
                    "true_label": [label],
                    "multi_label": False,
                }
            ]
        },
    }


def test_validate_record_accepts_a_classification_row():
    validate_record(_row("Please refund the duplicate charge.", label="yes"))


def test_validate_record_rejects_gold_outside_the_candidate_set():
    row = _row("hello", label="maybe")
    with pytest.raises(RecordError, match="not in the candidate set"):
        validate_record(row)


def test_validate_record_rejects_empty_input():
    row = _row("   ")
    with pytest.raises(RecordError, match="non-empty"):
        validate_record(row)


def test_split_is_deterministic_and_disjoint():
    grouped = {
        "alpha": [_row(f"alpha {i}", label="yes" if i % 2 == 0 else "no") for i in range(10)],
        "beta": [_row(f"beta {i}", task="route", label="a", labels=["a", "b"]) for i in range(5)],
    }
    first_train, first_eval = split_by_domain(grouped, eval_ratio=0.2, seed=42)
    second_train, second_eval = split_by_domain(grouped, eval_ratio=0.2, seed=42)
    assert [row["input"] for row in first_train] == [row["input"] for row in second_train]
    assert [row["input"] for row in first_eval] == [row["input"] for row in second_eval]
    train_ids = {row["input"] for row in first_train}
    eval_ids = {row["input"] for row in first_eval}
    assert train_ids.isdisjoint(eval_ids)
    assert len(first_eval) == 3
    assert len(first_train) == 12


def test_jsonl_round_trip(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [_row("A ticket about billing.", label="yes")])
    loaded = read_jsonl(path)
    assert loaded[0]["output"]["classifications"][0]["true_label"] == ["yes"]
    assert json.loads(path.read_text().splitlines()[0])["input"].startswith("A ticket")
