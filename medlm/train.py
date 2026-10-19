#!/usr/bin/env python3
"""Pretrain MedLM from random initialization (Mac MPS / CPU).

Example (after corpus + tokenizer + encode):

    python -m medlm.train --max-steps 500 --out models/medlm-120m

Not a medical device. Research pretraining only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medlm.config import MedLMConfig
from medlm.data import TokenDataset
from medlm.model import MedLM


def pick_device(name: str | None) -> str:
    if name:
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_lr(step: int, config: MedLMConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    if step >= config.max_steps:
        return config.learning_rate * 0.1
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.learning_rate * (0.1 + 0.9 * coeff)


@torch.no_grad()
def estimate_loss(model: MedLM, loader: DataLoader, device: str, batches: int = 20) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= batches:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/medlm")
    parser.add_argument("--out", default="models/medlm-120m")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="checkpoint dir to resume")
    args = parser.parse_args()

    config = MedLMConfig(data_dir=args.data_dir, out_dir=args.out, seed=args.seed)
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.grad_accum is not None:
        config.grad_accum = args.grad_accum
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.block_size is not None:
        config.block_size = args.block_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.save_every is not None:
        config.save_every = args.save_every
    if args.eval_every is not None:
        config.eval_every = args.eval_every
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps

    torch.manual_seed(config.seed)
    device = pick_device(args.device)

    data_dir = Path(config.data_dir)
    tok_path = Path(config.tokenizer_path)
    if tok_path.is_file():
        from tokenizers import Tokenizer

        vocab = Tokenizer.from_file(str(tok_path)).get_vocab_size()
        config.vocab_size = vocab
        print(f"tokenizer vocab_size={vocab}")

    print(f"device={device}  config≈{config.param_count_estimate()/1e6:.1f}M (estimate)")

    train_ds = TokenDataset(data_dir / "train.bin", config.block_size)
    val_ds = TokenDataset(data_dir / "val.bin", config.block_size)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, drop_last=True)

    if args.resume:
        model = MedLM.from_pretrained(args.resume, map_location="cpu")
        # keep architecture from checkpoint; re-apply CLI train hyperparams
        arch = model.config
        config.vocab_size = arch.vocab_size
        config.n_layer = arch.n_layer
        config.n_head = arch.n_head
        config.n_embd = arch.n_embd
        config.block_size = arch.block_size
        config.dropout = arch.dropout
        config.bias = arch.bias
        model.config = config
        print(f"resumed weights from {args.resume}")
    else:
        model = MedLM(config)
    model.to(device)
    n_params = model.num_parameters()
    print(f"parameters={n_params/1e6:.2f}M  block={config.block_size}  "
          f"batch={config.batch_size}x{config.grad_accum}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
        fused=False,
    )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save(out_dir / "config.json")
    # copy tokenizer next to model for inference
    tok_src = Path(config.tokenizer_path)
    if tok_src.is_file():
        (out_dir / "tokenizer.json").write_bytes(tok_src.read_bytes())

    model.train()
    step = 0
    running = 0.0
    t0 = time.time()
    train_iter = iter(train_loader)

    while step < config.max_steps:
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _micro in range(config.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            loss = loss / config.grad_accum
            loss.backward()
            loss_accum += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.step()

        running += loss_accum
        step += 1

        if step % 20 == 0 or step == 1:
            dt = time.time() - t0
            print(
                f"step {step}/{config.max_steps}  loss={loss_accum:.4f}  "
                f"avg20={running/min(step,20):.4f}  lr={lr:.2e}  "
                f"{dt:.1f}s",
                flush=True,
            )
            if step % 20 == 0:
                running = 0.0
                t0 = time.time()

        if step % config.eval_every == 0 or step == config.max_steps:
            val_loss = estimate_loss(model, val_loader, device)
            print(f"eval step={step}  val_loss={val_loss:.4f}", flush=True)
            (out_dir / "train_log.jsonl").open("a").write(
                json.dumps({"step": step, "val_loss": val_loss, "train_loss": loss_accum}) + "\n"
            )

        if step % config.save_every == 0 or step == config.max_steps:
            # single checkpoint dir only — Mac disk is tight (~model.pt is ~0.5 GB)
            model.save_pretrained(out_dir)
            print(f"saved {out_dir} @ step {step}", flush=True)

    manifest = {
        "name": "MedLM-120M",
        "parameters": n_params,
        "steps": config.max_steps,
        "device": device,
        "disclaimer": "Research LM from random init on compact healthcare-style English. Not a medical device.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("done", json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
