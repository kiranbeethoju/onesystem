"""Train OneSystem end to end and save the full model under its own name.

Steps
-----
1. Download the public development data and cut train / calibration / eval
   slices per domain.
2. Initialise the network from a general-purpose text encoder.
3. Score the untrained network on the eval slice (the zero-shot baseline).
4. Fine-tune the whole network with cross-entropy over each head's candidate
   set (sigmoid BCE for multi-label heads).
5. Fit one confidence temperature on the calibration slice.
6. Score the eval slice again and save ``models/onesystem``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from onesystem import DATASET_ID, DEFAULT_ENCODER, MODEL_NAME, __version__
from onesystem.data import CALIB_PATH, EVAL_PATH, TRAIN_PATH, prepare_splits, read_jsonl
from onesystem.hub import LOCAL_MODEL_DIR

MANIFEST_FILE = "onesystem.json"


@dataclass
class TrainConfig:
    encoder: str = DEFAULT_ENCODER
    epochs: int = 8
    batch_size: int = 16
    encoder_lr: float = 3e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    max_text_len: int = 8192
    max_label_len: int = 128
    # For plain LMs (e.g. ModernBERT) set >0. Embedding backbones (GTE/Jina/BGE) use 0.
    freeze_encoder_epochs: int = 0
    seed: int = 42
    output_dir: str = str(LOCAL_MODEL_DIR)
    skip_zero_shot: bool = False


# ------------------------------------------------------------------ examples
@dataclass
class Pair:
    task: str
    text: str
    labels: List[str]
    gold: List[str]
    multi_label: bool


def pairs_from_records(records: List[Dict]) -> List[Pair]:
    pairs = []
    for record in records:
        for head in record["output"]["classifications"]:
            gold = head["true_label"]
            gold = [gold] if isinstance(gold, str) else list(gold)
            pairs.append(
                Pair(
                    task=head["task"],
                    text=record["input"],
                    labels=list(head["labels"]),
                    gold=gold,
                    multi_label=bool(head.get("multi_label")) or len(gold) > 1,
                )
            )
    return pairs


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------ training
def batch_loss(network, tokenizer, batch: List[Pair], device):
    import torch
    import torch.nn.functional as F

    from onesystem.modeling import tokenize

    text_batch = tokenize(tokenizer, [network.text_prompt(p.task, p.text) for p in batch], network.config.max_text_len, device)
    text_embeddings = network.encode_text(**text_batch)

    label_keys: List[tuple] = []
    index_of: Dict[tuple, int] = {}
    for pair in batch:
        for label in pair.labels:
            key = (pair.task, label)
            if key not in index_of:
                index_of[key] = len(label_keys)
                label_keys.append(key)
    label_batch = tokenize(tokenizer, [network.label_prompt(task, label) for task, label in label_keys], network.config.max_label_len, device)
    label_embeddings = network.encode_labels(**label_batch)
    all_scores = network.scores(text_embeddings, label_embeddings)

    total = text_embeddings.new_zeros(())
    for row, pair in enumerate(batch):
        columns = torch.tensor([index_of[(pair.task, label)] for label in pair.labels], device=device)
        logits = all_scores[row, columns]
        if pair.multi_label:
            target = torch.tensor([1.0 if label in pair.gold else 0.0 for label in pair.labels], device=device)
            total = total + F.binary_cross_entropy_with_logits(logits, target)
        else:
            target = torch.tensor(pair.labels.index(pair.gold[0]), device=device)
            total = total + F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
    return total / len(batch)


def _set_encoder_trainable(network, trainable: bool) -> None:
    for name, param in network.named_parameters():
        if name.startswith("encoder."):
            param.requires_grad = trainable


def train(network, tokenizer, pairs: List[Pair], config: TrainConfig, device: str) -> Dict:
    import torch

    network.train()
    head_params = [p for n, p in network.named_parameters() if not n.startswith("encoder.")]
    encoder_params = [p for n, p in network.named_parameters() if n.startswith("encoder.")]
    freeze_epochs = max(0, int(config.freeze_encoder_epochs))
    if freeze_epochs:
        _set_encoder_trainable(network, False)
        print(f"freezing encoder for first {freeze_epochs} epoch(s)", flush=True)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config.encoder_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(pairs) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    warmup = max(1, int(total_steps * config.warmup_ratio))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        remaining = max(1, total_steps - warmup)
        return max(0.0, (total_steps - step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    history = []
    step = 0
    started = time.time()
    for epoch in range(config.epochs):
        if freeze_epochs and epoch == freeze_epochs:
            _set_encoder_trainable(network, True)
            print(f"unfreezing encoder at epoch {epoch + 1} (lr {config.encoder_lr})", flush=True)
        order = list(range(len(pairs)))
        random.shuffle(order)
        running = 0.0
        for start in range(0, len(order), config.batch_size):
            batch = [pairs[i] for i in order[start:start + config.batch_size]]
            loss = batch_loss(network, tokenizer, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            loss_value = loss.item()
            running += loss_value
            if step % 20 == 0:
                elapsed = time.time() - started
                print(f"epoch {epoch + 1}/{config.epochs} step {step}/{total_steps} loss {loss_value:.4f} {elapsed:.0f}s", flush=True)
        epoch_loss = running / steps_per_epoch
        history.append({"epoch": epoch + 1, "loss": round(epoch_loss, 4)})
        print(f"epoch {epoch + 1} mean loss {epoch_loss:.4f}", flush=True)
    network.eval()
    return {"steps": total_steps, "history": history, "seconds": round(time.time() - started, 1)}


# ------------------------------------------------------------------ manifest
def build_manifest(config: TrainConfig, counts: Dict, extra: Dict) -> Dict:
    return {
        "name": MODEL_NAME,
        "version": __version__,
        "model_type": "onesystem",
        "encoder_init": config.encoder,
        "dataset": DATASET_ID,
        "dataset_split": "development",
        "holdout": "per domain: 20 percent eval, 10 percent calibration, rest train",
        "training": {k: v for k, v in asdict(config).items() if k not in {"output_dir", "skip_zero_shot"}},
        **counts,
        **extra,
    }


def write_manifest(document: Dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_FILE).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ main
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train OneSystem")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-text-len", type=int, default=8192, help="text tokens including task prefix; max depends on --encoder")
    parser.add_argument("--max-label-len", type=int, default=128, help="label tokens including task prefix / description")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0, help="train head only for N epochs before unfreezing encoder")
    parser.add_argument("--encoder-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default=str(LOCAL_MODEL_DIR))
    parser.add_argument("--skip-zero-shot", action="store_true")
    args = parser.parse_args(argv)
    from onesystem import ENCODER_MAX_POSITIONS

    if args.max_text_len > ENCODER_MAX_POSITIONS:
        raise SystemExit(
            f"--max-text-len {args.max_text_len} exceeds encoder ceiling "
            f"{ENCODER_MAX_POSITIONS}. Use a longer-context --encoder to go beyond that."
        )
    if args.max_label_len > ENCODER_MAX_POSITIONS:
        raise SystemExit(f"--max-label-len {args.max_label_len} exceeds encoder ceiling {ENCODER_MAX_POSITIONS}")
    config = TrainConfig(
        encoder=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_text_len=args.max_text_len,
        max_label_len=args.max_label_len,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        output_dir=args.output_dir,
        skip_zero_shot=args.skip_zero_shot,
    )

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    from onesystem.evaluate import fit_temperature, score_records
    from onesystem.model import OneSystem
    from onesystem.modeling import OneSystemModel

    seed_everything(config.seed)
    counts = prepare_splits()
    print(f"splits: {counts}", flush=True)
    train_records = read_jsonl(TRAIN_PATH)
    calib_records = read_jsonl(CALIB_PATH)
    eval_records = read_jsonl(EVAL_PATH)

    device = pick_device()
    print(f"initialising {MODEL_NAME} from {config.encoder} on {device}", flush=True)
    network = OneSystemModel.from_encoder_name(
        config.encoder,
        name=MODEL_NAME,
        version=__version__,
        max_text_len=config.max_text_len,
        max_label_len=config.max_label_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.encoder, trust_remote_code=True)
    network.to(device)

    extra: Dict = {}
    if not config.skip_zero_shot:
        zero_shot = score_records(OneSystem.from_parts(network, tokenizer, device), eval_records)
        extra["zero_shot_eval"] = {"exact_match": zero_shot["exact_match"], "ece": zero_shot["ece"]}
        print(f"zero-shot eval exact match {zero_shot['exact_match']:.3f}", flush=True)

    pairs = pairs_from_records(train_records)
    print(f"training on {len(pairs)} decision heads", flush=True)
    extra["train_run"] = train(network, tokenizer, pairs, config, device)

    model = OneSystem.from_parts(network, tokenizer, device)
    temperature, calib_nll = fit_temperature(model, calib_records)
    network.config.temperature = temperature
    model.clear_cache()
    extra["calibration"] = {"temperature": round(temperature, 4), "calib_nll": round(calib_nll, 4)}
    print(f"temperature {temperature:.3f}", flush=True)

    final = score_records(model, eval_records)
    extra["eval"] = {"exact_match": final["exact_match"], "ece": final["ece"], "heads": final["heads"], "correct": final["correct"]}
    print(f"eval exact match {final['exact_match']:.3f} ece {final['ece']:.3f}", flush=True)

    output_dir = Path(config.output_dir)
    network.save(output_dir, tokenizer=tokenizer)
    write_manifest(build_manifest(config, counts, extra), output_dir)
    (output_dir / "eval.json").write_text(json.dumps({**final, "note": (
        "Scored on a holdout cut from the public development split. "
        "This is not the held-out fast-decisions test benchmark."
    )}, indent=2) + "\n", encoding="utf-8")
    print(f"saved {MODEL_NAME} to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
