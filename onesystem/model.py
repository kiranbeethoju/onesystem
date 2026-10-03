"""Inference API for OneSystem."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from onesystem import MODEL_NAME
from onesystem.hub import resolve

LabelSpec = Union[Sequence[str], Dict]


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalise_tasks(tasks: Dict) -> Dict[str, Dict]:
    """Accept ``{"task": [labels]}`` or ``{"task": {"labels": [...], ...}}``."""
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("tasks must be a non-empty dictionary of label sets")
    cleaned: Dict[str, Dict] = {}
    for task, spec in tasks.items():
        if isinstance(spec, dict):
            labels = spec.get("labels")
            descriptions = spec.get("descriptions") or {}
            if isinstance(labels, dict):
                descriptions = {**labels, **descriptions}
                labels = list(labels)
            multi_label = bool(spec.get("multi_label", False))
            threshold = float(spec.get("threshold", 0.5))
            ordinal = spec.get("ordinal")
        else:
            labels = spec
            descriptions = {}
            multi_label = False
            threshold = 0.5
            ordinal = None
        if not isinstance(labels, (list, tuple)) or len(labels) < 2:
            raise ValueError(f"task {task!r} needs at least two labels")
        labels = [str(label) for label in labels]
        if len(set(labels)) != len(labels):
            raise ValueError(f"task {task!r} has duplicate labels")
        cleaned[str(task)] = {
            "labels": labels,
            "descriptions": {str(k): str(v) for k, v in descriptions.items()},
            "multi_label": multi_label,
            "threshold": threshold,
            "ordinal": ordinal,
        }
    return cleaned


class OneSystem:
    """Load OneSystem weights and make typed decisions.

    >>> model = OneSystem.load()
    >>> model.classify("Please refund the duplicate charge.", {"intent": ["refund", "other"]})
    """

    def __init__(self, network, tokenizer, device: str, name: str = MODEL_NAME):
        self.network = network
        self.tokenizer = tokenizer
        self.device = device
        self.name = name
        self._label_cache: Dict[Tuple[str, str, str], "object"] = {}

    @classmethod
    def load(cls, source: Optional[str] = None, device: Optional[str] = None) -> "OneSystem":
        """``source`` is a directory, ``github:owner/repo@tag``, or ``None`` for the default."""
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoTokenizer

        from onesystem.modeling import OneSystemModel

        directory = resolve(source)
        device = device or default_device()
        network = OneSystemModel.from_directory(Path(directory), device=device)
        tokenizer = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=True)
        return cls(network, tokenizer, device, name=network.config.name)

    @classmethod
    def from_parts(cls, network, tokenizer, device: str) -> "OneSystem":
        network.to(device)
        network.eval()
        return cls(network, tokenizer, device, name=network.config.name)

    # --------------------------------------------------------------- caching
    def clear_cache(self) -> None:
        self._label_cache.clear()

    def _label_embeddings(self, task: str, labels: List[str], descriptions: Dict[str, str]):
        import torch

        from onesystem.modeling import tokenize

        keys = [(task, label, descriptions.get(label, "")) for label in labels]
        missing = [key for key in keys if key not in self._label_cache]
        if missing:
            prompts = [self.network.label_prompt(task, label, desc or None) for task, label, desc in missing]
            batch = tokenize(self.tokenizer, prompts, self.network.config.max_label_len, self.device)
            with torch.no_grad():
                embeddings = self.network.encode_labels(**batch)
            for key, embedding in zip(missing, embeddings):
                self._label_cache[key] = embedding
        return torch.stack([self._label_cache[key] for key in keys])

    # ------------------------------------------------------------- inference
    def classify(self, text: str, tasks: Dict, min_confidence: Optional[float] = None) -> Dict:
        return self.classify_batch([text], tasks, min_confidence=min_confidence)[0]

    def classify_batch(self, texts: Sequence[str], tasks: Dict, min_confidence: Optional[float] = None) -> List[Dict]:
        import torch

        from onesystem.modeling import expected_level, labels_look_ordinal, tokenize

        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("every text must be a non-empty string")
        specs = normalise_tasks(tasks)
        temperature = float(self.network.config.temperature)
        results: List[Dict] = [{} for _ in texts]
        for task, spec in specs.items():
            prompts = [self.network.text_prompt(task, text) for text in texts]
            batch = tokenize(self.tokenizer, prompts, self.network.config.max_text_len, self.device)
            with torch.no_grad():
                text_embeddings = self.network.encode_text(**batch)
                label_embeddings = self._label_embeddings(task, spec["labels"], spec["descriptions"])
                logits = self.network.scores(text_embeddings, label_embeddings) / temperature
                if spec["multi_label"]:
                    probabilities = torch.sigmoid(logits)
                else:
                    probabilities = torch.softmax(logits, dim=-1)
            probabilities = probabilities.cpu().tolist()
            ordinal = spec["ordinal"]
            if ordinal is None:
                ordinal = labels_look_ordinal(spec["labels"])
            for index, row in enumerate(probabilities):
                results[index][task] = self._format(spec, row, ordinal, min_confidence, expected_level)
        return results

    @staticmethod
    def _format(spec: Dict, row: List[float], ordinal: bool, min_confidence: Optional[float], expected_level) -> Dict:
        labels = spec["labels"]
        table = {label: round(float(p), 4) for label, p in zip(labels, row)}
        if spec["multi_label"]:
            chosen = [label for label, p in zip(labels, row) if p >= spec["threshold"]]
            return {"labels": chosen, "probabilities": table, "threshold": spec["threshold"]}
        best = max(range(len(labels)), key=lambda i: row[i])
        answer = {
            "label": labels[best],
            "confidence": round(float(row[best]), 4),
            "probabilities": table,
        }
        if ordinal:
            answer["score"] = round(expected_level(row), 4)
        if min_confidence is not None and row[best] < min_confidence:
            answer["abstain"] = True
            answer["label"] = None
        return answer
