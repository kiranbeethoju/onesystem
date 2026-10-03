"""Fine-tune a OneSystem LoRA adapter on the public development split."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from onesystem import BASE_CHECKPOINT, DATASET_ID, MODEL_NAME, __version__
from onesystem.data import EVAL_PATH, TRAIN_PATH, prepare_splits
from onesystem.model import ADAPTER_DIR

OUTPUT_DIR = Path("output/onesystem")
MANIFEST_PATH = ADAPTER_DIR / "onesystem.json"


def build_training_config(output_dir: Path = OUTPUT_DIR):
    """LoRA settings that fit a single Apple GPU or a CPU fallback."""
    from gliner2.training.trainer import TrainingConfig

    return TrainingConfig(
        output_dir=str(output_dir),
        experiment_name=MODEL_NAME,
        num_epochs=1,
        batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=4,
        encoder_lr=1e-5,
        task_lr=5e-4,
        warmup_ratio=0.06,
        fp16=False,
        bf16=False,
        eval_strategy="no",
        save_total_limit=1,
        logging_steps=10,
        num_workers=0,
        pin_memory=False,
        seed=42,
        max_len=384,
        use_lora=True,
        lora_r=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
        lora_target_modules=["encoder", "classifier"],
        save_adapter_only=True,
        fused_optimizer=False,
        compile_model=False,
        gradient_checkpointing=False,
        group_by_length=False,
    )


def build_manifest(counts: dict, adapter_dir: Path = ADAPTER_DIR) -> dict:
    return {
        "name": MODEL_NAME,
        "version": __version__,
        "base_checkpoint": BASE_CHECKPOINT,
        "dataset": DATASET_ID,
        "dataset_split": "development",
        "holdout": "20 percent of each domain inside the public development split",
        "method": "lora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_target_modules": ["encoder", "classifier"],
        "epochs": 1,
        "max_len": 384,
        "seed": 42,
        "adapter_dir": str(adapter_dir),
        **counts,
    }


def place_on_best_device(trainer) -> str:
    """The upstream trainer selects CUDA or CPU. Prefer MPS when it exists."""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    trainer.device = device
    trainer.model.to(device)
    return device.type


def save_named_adapter(output_dir: Path = OUTPUT_DIR, adapter_dir: Path = ADAPTER_DIR) -> Path:
    source = output_dir / "final"
    if not (source / "adapter_config.json").is_file():
        raise FileNotFoundError(f"trained adapter not found at {source}")
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    shutil.copytree(source, adapter_dir)
    return adapter_dir


def write_manifest(document: dict, path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    counts = prepare_splits()
    print(
        f"prepared {counts['domains']} domains: "
        f"{counts['train_rows']} train rows, {counts['eval_rows']} eval rows",
        flush=True,
    )

    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer

    print(f"loading {BASE_CHECKPOINT}", flush=True)
    model = AutoExtractor.from_pretrained(BASE_CHECKPOINT)
    config = build_training_config()
    trainer = ExtractorTrainer(model, config)
    device = place_on_best_device(trainer)
    print(f"training {MODEL_NAME} on {device}", flush=True)
    trainer.train(train_data=str(TRAIN_PATH))
    adapter_dir = save_named_adapter()
    write_manifest(build_manifest(counts, adapter_dir))
    print(f"saved {MODEL_NAME} adapter to {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
