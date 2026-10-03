"""The OneSystem network: encoder, label-scoring head, calibration.

Design
------
* The document is encoded once per decision head, conditioned on the head's
  task name: ``"{task}: {text}"``.
* Every candidate label is encoded separately: ``"{task}: {label}"`` (plus an
  optional description). Label embeddings depend only on the schema, so they
  are cached. A fixed schema therefore costs one text pass per head, however
  many labels it has.
* Text and labels share one linear projection that starts as the identity,
  so the untrained network already scores by the encoder's own similarity.
* Scores are scaled cosine similarities. Single-label heads take a softmax
  over the candidate set; multi-label heads take a sigmoid per label.
* A scalar temperature, fit on a calibration split after training, divides
  the logits at inference so the reported confidence is a probability you can
  branch on.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


@dataclass
class OneSystemConfig:
    name: str = "OneSystem"
    version: str = "0.2.1"
    model_type: str = "onesystem"
    encoder_name: str = ""
    encoder_config: Dict = field(default_factory=dict)
    hidden_size: int = 768
    max_text_len: int = 512
    max_label_len: int = 64
    text_template: str = "{task}: {text}"
    label_template: str = "{task}: {label}"
    described_label_template: str = "{task}: {label}. {description}"
    temperature: float = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "OneSystemConfig":
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / CONFIG_FILE).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "OneSystemConfig":
        return cls.from_dict(json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8")))


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def _identity_projection(hidden_size: int) -> nn.Linear:
    """A linear map that starts as the identity.

    Text and labels share this map, so at initialisation the scores are the
    encoder's own cosine similarities and training only has to sharpen them.
    Two independent random projections would put text and labels in unrelated
    spaces and throw the pretrained geometry away.
    """
    layer = nn.Linear(hidden_size, hidden_size)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(hidden_size))
        layer.bias.zero_()
    return layer


def _clean_encoder_config(config: Dict) -> Dict:
    dropped = {"_name_or_path", "transformers_version", "architectures", "torch_dtype", "dtype"}
    return {key: value for key, value in config.items() if key not in dropped}


class OneSystemModel(nn.Module):
    """Encoder plus scoring head. Use :class:`onesystem.model.OneSystem` for inference."""

    def __init__(self, config: OneSystemConfig, encoder: nn.Module):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.proj = _identity_projection(config.hidden_size)
        # Learned logit scale, initialised to 1 / 0.05 the way contrastive encoders do.
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        # Multi-label heads use a sigmoid per label; encoder cosines sit around
        # 0.65, so a learned offset centres the decision boundary.
        self.multi_label_bias = nn.Parameter(torch.tensor(-13.0))

    # ------------------------------------------------------------- construction
    @classmethod
    def from_encoder_name(cls, encoder_name: str, **overrides) -> "OneSystemModel":
        """Build a fresh network for training from a Hugging Face encoder.

        Eager attention is used so that dropout works during training on
        every backend; fused scaled-dot-product attention has no dropout
        kernel on Apple MPS and silently degrades there.
        """
        from transformers import AutoConfig, AutoModel

        encoder = AutoModel.from_pretrained(encoder_name, add_pooling_layer=False, attn_implementation="eager")
        encoder_config = _clean_encoder_config(AutoConfig.from_pretrained(encoder_name).to_dict())
        config = OneSystemConfig(
            encoder_name=encoder_name,
            encoder_config=encoder_config,
            hidden_size=int(encoder.config.hidden_size),
            **overrides,
        )
        return cls(config, encoder)

    @classmethod
    def from_directory(cls, directory: Path, device: Optional[str] = None) -> "OneSystemModel":
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel

        config = OneSystemConfig.load(directory)
        encoder_config = dict(config.encoder_config)
        model_type = encoder_config.pop("model_type")
        hf_config = AutoConfig.for_model(model_type, **encoder_config)
        encoder = AutoModel.from_config(hf_config, add_pooling_layer=False)
        model = cls(config, encoder)
        state = load_file(str(directory / WEIGHTS_FILE))
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [key for key in missing if not key.startswith("encoder.embeddings.position_ids")]
        if missing or unexpected:
            raise RuntimeError(f"weights do not match the model: missing={missing[:5]} unexpected={unexpected[:5]}")
        if device:
            model.to(device)
        model.eval()
        return model

    def save(self, directory: Path, tokenizer=None) -> None:
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save(directory)
        state = {key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()}
        save_file(state, str(directory / WEIGHTS_FILE), metadata={"format": "pt", "name": self.config.name})
        if tokenizer is not None:
            tokenizer.save_pretrained(str(directory))

    # --------------------------------------------------------------- encoding
    def _pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return masked_mean(output.last_hidden_state, attention_mask)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self._pool(input_ids, attention_mask)), dim=-1)

    def encode_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self._pool(input_ids, attention_mask)), dim=-1)

    def scores(self, text_embeddings: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
        """Raw logits for every (text, label) pair: ``[n_text, n_label]``."""
        return (text_embeddings @ label_embeddings.T) * self.log_scale.exp()

    # ---------------------------------------------------------------- prompts
    def text_prompt(self, task: str, text: str) -> str:
        return self.config.text_template.format(task=task, text=text)

    def label_prompt(self, task: str, label: str, description: Optional[str] = None) -> str:
        if description:
            return self.config.described_label_template.format(task=task, label=label, description=description)
        return self.config.label_template.format(task=task, label=label)


def tokenize(tokenizer, prompts: Sequence[str], max_len: int, device) -> Dict[str, torch.Tensor]:
    batch = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}


def expected_level(probabilities: Sequence[float]) -> float:
    """Expected rank for an ordinal head whose labels are ordered low to high."""
    return float(sum(index * p for index, p in enumerate(probabilities)))


def labels_look_ordinal(labels: List[str]) -> bool:
    return len(labels) >= 2 and all(label.strip().lstrip("-").isdigit() for label in labels)
