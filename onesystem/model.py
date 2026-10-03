"""Load the OneSystem adapter on top of the public Decide checkpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union

from onesystem import BASE_CHECKPOINT, MODEL_NAME

ADAPTER_DIR = Path("models/onesystem")


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OneSystem:
    """Schema-conditioned classifier.

    Pass the candidate labels at call time. The model returns the chosen label
    and, when requested, a confidence score. It does not generate text.
    """

    def __init__(self, model, name: str = MODEL_NAME):
        self.model = model
        self.name = name

    @classmethod
    def load(
        cls,
        adapter_dir: Union[str, Path, None] = ADAPTER_DIR,
        base_checkpoint: str = BASE_CHECKPOINT,
        device: Optional[str] = None,
    ) -> "OneSystem":
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from gliner2 import AutoExtractor

        model = AutoExtractor.from_pretrained(base_checkpoint)
        adapter = Path(adapter_dir) if adapter_dir else None
        if adapter and (adapter / "adapter_config.json").is_file():
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter))
        model.to(device or default_device())
        return cls(model)

    def classify(self, text: str, tasks: Dict, include_confidence: bool = True) -> Dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not isinstance(tasks, dict) or not tasks:
            raise ValueError("tasks must be a non-empty dictionary of label sets")
        return self.model.classify_text(text, tasks, include_confidence=include_confidence)
