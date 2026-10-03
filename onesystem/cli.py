"""Classify one text with OneSystem from the command line."""

from __future__ import annotations

import argparse
import json

from onesystem.model import OneSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a typed decision with OneSystem")
    parser.add_argument("text", help="the document or message to classify")
    parser.add_argument("--task", required=True, help="name of the decision, such as intent")
    parser.add_argument("--labels", required=True, help="comma-separated candidate labels")
    parser.add_argument("--multi-label", action="store_true", help="return every label above the threshold")
    parser.add_argument("--min-confidence", type=float, default=None, help="abstain below this probability")
    parser.add_argument("--source", default=None, help="model directory or github:owner/repo@tag")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu")
    args = parser.parse_args()

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if len(labels) < 2:
        parser.error("--labels needs at least two comma-separated labels")
    model = OneSystem.load(args.source, device=args.device)
    tasks = {args.task: {"labels": labels, "multi_label": args.multi_label}}
    result = model.classify(args.text, tasks, min_confidence=args.min_confidence)
    print(json.dumps({"model": model.name, "answers": result}, indent=2))


if __name__ == "__main__":
    main()
