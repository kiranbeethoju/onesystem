"""Load and split the public fast-decisions development set.

The Hugging Face release is a development split of 1,700 rows. OneSystem keeps
a per-domain holdout inside that split so training and evaluation do not share
rows. That holdout is not Fastino's unpublished test set.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from onesystem import DATASET_ID

RAW_DIR = Path("data/raw")
TRAIN_PATH = Path("data/train.jsonl")
EVAL_PATH = Path("data/eval.jsonl")


class RecordError(ValueError):
    """A training row is missing a field the trainer needs."""


def validate_record(record: Mapping) -> None:
    """Raise RecordError unless this row is a classification example."""
    if not isinstance(record, Mapping):
        raise RecordError("a record must be an object")
    text = record.get("input")
    if not isinstance(text, str) or not text.strip():
        raise RecordError("input must be a non-empty string")
    output = record.get("output")
    if not isinstance(output, Mapping):
        raise RecordError("output must be an object")
    heads = output.get("classifications")
    if not isinstance(heads, list) or not heads:
        raise RecordError("output.classifications must be a non-empty list")
    for index, head in enumerate(heads):
        if not isinstance(head, Mapping):
            raise RecordError(f"classifications[{index}] must be an object")
        task = head.get("task")
        labels = head.get("labels")
        truth = head.get("true_label")
        if not isinstance(task, str) or not task:
            raise RecordError(f"classifications[{index}].task must be a string")
        if not isinstance(labels, list) or not labels or not all(isinstance(item, str) for item in labels):
            raise RecordError(f"classifications[{index}].labels must be a list of strings")
        if isinstance(truth, str):
            truth_list = [truth]
        elif isinstance(truth, list) and truth and all(isinstance(item, str) for item in truth):
            truth_list = truth
        else:
            raise RecordError(f"classifications[{index}].true_label must be a string or a list of strings")
        missing = [item for item in truth_list if item not in labels]
        if missing:
            raise RecordError(
                f"classifications[{index}] gold label(s) {missing} are not in the candidate set"
            )
        if len(truth_list) > 1 and not head.get("multi_label"):
            raise RecordError(
                f"classifications[{index}] has several gold labels but multi_label is false"
            )


def training_payload(record: Mapping) -> Dict:
    """Return the input/output object the trainer accepts."""
    validate_record(record)
    return {"input": record["input"], "output": record["output"]}


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record)
            except (json.JSONDecodeError, RecordError) as exc:
                raise RecordError(f"{path}:{line_number}: {exc}") from exc
            rows.append(record)
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(training_payload(record), ensure_ascii=False) + "\n")


def split_by_domain(
    rows_by_domain: Mapping[str, Sequence[Mapping]],
    eval_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[List[Dict], List[Dict]]:
    """Hold out the same fraction of every domain, with a fixed shuffle."""
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1")
    rng = random.Random(seed)
    train: List[Dict] = []
    heldout: List[Dict] = []
    for domain in sorted(rows_by_domain):
        rows = [dict(row) for row in rows_by_domain[domain]]
        rng.shuffle(rows)
        n_eval = int(len(rows) * eval_ratio)
        if len(rows) > 1 and n_eval == 0:
            n_eval = 1
        if n_eval >= len(rows):
            n_eval = len(rows) - 1
        heldout.extend(rows[:n_eval])
        train.extend(rows[n_eval:])
    return train, heldout


def group_jsonl_files(paths: Iterable[Path]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for path in sorted(paths):
        grouped[path.stem] = read_jsonl(path)
    if not grouped:
        raise FileNotFoundError("no jsonl files found")
    return grouped


def download_raw(destination: Path = RAW_DIR) -> List[Path]:
    """Download the public development jsonl files."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        DATASET_ID,
        repo_type="dataset",
        local_dir=str(destination),
        allow_patterns=["*.jsonl"],
    )
    files = sorted(destination.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no jsonl files downloaded into {destination}")
    return files


def prepare_splits(
    raw_dir: Path = RAW_DIR,
    train_path: Path = TRAIN_PATH,
    eval_path: Path = EVAL_PATH,
    eval_ratio: float = 0.2,
    seed: int = 42,
    download: bool = True,
) -> Dict[str, int]:
    """Write train and eval jsonl files. Returns row counts."""
    files = sorted(raw_dir.rglob("*.jsonl"))
    if not files and download:
        files = download_raw(raw_dir)
    grouped = group_jsonl_files(files)
    train, heldout = split_by_domain(grouped, eval_ratio=eval_ratio, seed=seed)
    write_jsonl(train_path, train)
    write_jsonl(eval_path, heldout)
    return {
        "domains": len(grouped),
        "train_rows": len(train),
        "eval_rows": len(heldout),
    }
