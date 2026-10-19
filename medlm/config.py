"""MedLM-120M config (~120M params, GPT-2-small class — fits 18 GB Mac MPS)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class MedLMConfig:
    # Architecture → ~120–130M with ~14k–32k vocab (16×768 GPT-style)
    vocab_size: int = 32000
    n_layer: int = 16
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 512
    dropout: float = 0.0
    bias: bool = False

    # Train (Mac / MPS defaults — conservative for ≤18 GB unified memory)
    batch_size: int = 1
    grad_accum: int = 16  # effective batch 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 5000
    eval_every: int = 100
    save_every: int = 250
    grad_clip: float = 1.0
    seed: int = 42

    # Paths
    data_dir: str = "data/medlm"
    out_dir: str = "models/medlm-120m"
    tokenizer_path: str = "data/medlm/tokenizer.json"

    def param_count_estimate(self) -> int:
        """Rough non-embedding + embedding param estimate."""
        V, C, L = self.vocab_size, self.n_embd, self.n_layer
        # token + pos emb, 12*L*(C^2) attn+mlp-ish, ln, lm_head tied → ~124M at these sizes
        emb = V * C + self.block_size * C
        per_layer = 12 * C * C  # rough
        return emb + L * per_layer + 2 * C

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedLMConfig":
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})

    def save(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
