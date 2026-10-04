"""Adapters that map each System-1 engine onto a common intent API."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
OPENJEV_SRC = Path(os.environ.get("OPENJEV_SRC", "/tmp/Verdict-open-jev"))
OPENJEV_MAX_OPTIONS = 24  # Verdict Choice schema maxItems


@dataclass
class IntentAnswer:
    label: Optional[str]
    confidence: float
    probabilities: Dict[str, float]
    abstain: bool = False
    error: Optional[str] = None
    meta: Optional[Dict] = None


class BaseAdapter:
    name: str = "base"
    track: str = "raw"

    def classify_intent(self, text: str, labels: Sequence[str]) -> IntentAnswer:
        raise NotImplementedError

    def warm(self, texts: Sequence[str], labels: Sequence[str], n: int = 3) -> None:
        for text in texts[:n]:
            self.classify_intent(text, labels)


class OneSystemAdapter(BaseAdapter):
    def __init__(self, source: Optional[str] = None, device: Optional[str] = None, track: str = "raw"):
        from onesystem.model import OneSystem

        self.name = "onesystem"
        self.track = track
        self.source = source
        self.model = OneSystem.load(source, device=device)

    def classify_intent(self, text: str, labels: Sequence[str]) -> IntentAnswer:
        try:
            out = self.model.classify(text, {"intent": list(labels)})["intent"]
            return IntentAnswer(
                label=out.get("label"),
                confidence=float(out.get("confidence") or 0.0),
                probabilities={k: float(v) for k, v in (out.get("probabilities") or {}).items()},
                abstain=bool(out.get("abstain")),
            )
        except Exception as exc:  # noqa: BLE001
            return IntentAnswer(label=None, confidence=0.0, probabilities={}, abstain=True, error=str(exc))


class LayaAdapter(BaseAdapter):
    def __init__(self, track: str = "raw"):
        from laya import Router

        self.name = "laya"
        self.track = track
        self.runner = Router()

    def classify_intent(self, text: str, labels: Sequence[str]) -> IntentAnswer:
        from laya import decide

        criteria = {str(lab): f"Utterance about: {lab.replace('_', ' ')}" for lab in labels}
        try:
            # Large candidate sets need a longer context budget (CLINC150 ≈ 150 options).
            n = len(labels)
            predict_kwargs = {}
            if n > 80:
                predict_kwargs["max_len"] = 2048
            elif n > 40:
                predict_kwargs["max_len"] = 1024
            details = decide(
                self.runner,
                text,
                questions={
                    "intent": {
                        "type": "choice",
                        "instructions": "Which intent best matches this utterance?",
                        "criteria": criteria,
                    }
                },
                return_details=True,
                **predict_kwargs,
            )
            ans = details.answers["intent"]
            probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
            return IntentAnswer(
                label=ans.get("choice"),
                confidence=float(ans.get("confidence") or ans.get("answer_confidence") or 0.0),
                probabilities=probs,
            )
        except Exception as exc:  # noqa: BLE001
            return IntentAnswer(label=None, confidence=0.0, probabilities={}, abstain=True, error=str(exc))


class OpenJevAdapter(BaseAdapter):
    """OpenJev (Verdict). Choice queries allow at most 24 options.

    For larger label sets the harness runs a sequential elimination tournament
    (documented in the comparison report). That is not identical to a single
    full-softmax over all labels.
    """

    def __init__(self, track: str = "raw", src: Optional[Path] = None, device: str = "cpu"):
        src = Path(src or OPENJEV_SRC)
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        os.chdir(src)  # relative artifact paths in engine
        from rlcd import DecisionEngine

        weights = src / "artifacts" / "v2"
        self.name = "openjev"
        self.track = track
        self.engine = DecisionEngine(model_name_or_path=str(weights), device=device)

    def _one_choice(self, text: str, labels: Sequence[str]):
        from rlcd import Choice, Option

        query = Choice(
            id="intent",
            question="Which intent best matches this utterance?",
            options=[Option(id=str(lab), description=str(lab).replace("_", " ")) for lab in labels],
        )
        batch = self.engine.evaluate(context=text, queries=[query])
        item = batch.results[0]
        probs = {k: float(v) for k, v in (item.probabilities or {}).items() if not str(k).startswith("__")}
        return item.selected_id, float(item.selected_probability or 0.0), probs, bool(item.is_abstention)

    def classify_intent(self, text: str, labels: Sequence[str]) -> IntentAnswer:
        labels = [str(x) for x in labels]
        try:
            if len(labels) <= OPENJEV_MAX_OPTIONS:
                lab, conf, probs, abstain = self._one_choice(text, labels)
                return IntentAnswer(label=lab, confidence=conf, probabilities=probs, abstain=abstain, meta={"mode": "single"})
            # Sequential tournament: keep champion, challenge with next unused labels.
            remaining = list(labels)
            champion = None
            champ_conf = 0.0
            champ_probs: Dict[str, float] = {}
            rounds = 0
            while remaining:
                if champion is None:
                    batch = remaining[:OPENJEV_MAX_OPTIONS]
                    remaining = remaining[OPENJEV_MAX_OPTIONS:]
                else:
                    room = OPENJEV_MAX_OPTIONS - 1
                    batch = [champion] + remaining[:room]
                    remaining = remaining[room:]
                lab, conf, probs, abstain = self._one_choice(text, batch)
                rounds += 1
                champion, champ_conf, champ_probs = lab, conf, probs
                if abstain:
                    return IntentAnswer(label=None, confidence=conf, probabilities=probs, abstain=True, meta={"mode": "tournament", "rounds": rounds})
            return IntentAnswer(
                label=champion,
                confidence=champ_conf,
                probabilities=champ_probs,
                meta={"mode": "tournament", "rounds": rounds, "n_labels": len(labels)},
            )
        except Exception as exc:  # noqa: BLE001
            return IntentAnswer(label=None, confidence=0.0, probabilities={}, abstain=True, error=str(exc))
