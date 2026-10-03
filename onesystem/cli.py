"""Classify one text with the saved OneSystem adapter."""

from __future__ import annotations

import argparse
import json

from onesystem.model import OneSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify text with OneSystem")
    parser.add_argument("text", help="the document or message to classify")
    parser.add_argument("--task", required=True, help="name of the decision, such as intent")
    parser.add_argument("--labels", required=True, help="comma-separated candidate labels")
    args = parser.parse_args()
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if len(labels) < 2:
        parser.error("--labels needs at least two comma-separated labels")
    model = OneSystem.load()
    result = model.classify(args.text, {args.task: labels})
    print(json.dumps({"model": model.name, "answers": result}, indent=2))


if __name__ == "__main__":
    main()
