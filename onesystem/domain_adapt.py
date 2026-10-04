"""Fine-tune a domain-specific OneSystem checkpoint.

Use this when you have labelled examples for *your* label set (support intents,
emotions, triage codes, …). The general v0.3.0 release is a strong starting
point for zero-shot / few-label schemas; domain adaptation is what pushes
accuracy on a fixed taxonomy (see BANKING77 / CLINC150 / HWU64 numbers).

Quick start
-----------
1. Write train (+ optional calib) JSONL in the OneSystem record format::

    {"input": "Please refund the duplicate charge.",
     "output": {"classifications": [{
         "task": "intent",
         "labels": ["refund", "status", "other"],
         "true_label": ["refund"],
         "multi_label": false
     }]}}

2. Train::

    python -m onesystem.domain_adapt \\
        --train data/my_domain/train.jsonl \\
        --output models/onesystem-mydomain \\
        --name OneSystem-MyDomain \\
        --epochs 4

3. Load and use::

    from onesystem.model import OneSystem
    model = OneSystem.load("models/onesystem-mydomain")
    model.classify("…", {"intent": ["refund", "status", "other"]})

If ``--calib`` is omitted, 10% of ``--train`` is held out for temperature fit
(stratified by gold label when possible). Official test sets must stay out of
both files.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from onesystem import DEFAULT_ENCODER, __version__
from onesystem.data import read_jsonl, write_jsonl
from onesystem.train import TrainConfig, pairs_from_records, pick_device, seed_everything, train


def _gold_key(record: Dict) -> str:
    head = record["output"]["classifications"][0]
    truth = head["true_label"]
    if isinstance(truth, str):
        return truth
    return "|".join(sorted(str(x) for x in truth))


def split_records_for_calib(
    records: Sequence[Dict],
    calib_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Hold out a calib slice; prefer stratified cut when heads share a gold key."""
    import random

    by_key: Dict[str, List[Dict]] = defaultdict(list)
    for row in records:
        by_key[_gold_key(row)].append(row)
    rng = random.Random(seed)
    train_rows: List[Dict] = []
    calib_rows: List[Dict] = []
    for key, items in sorted(by_key.items()):
        items = list(items)
        rng.shuffle(items)
        if len(items) == 1:
            train_rows.extend(items)
            continue
        n_calib = max(1, int(round(len(items) * calib_ratio)))
        calib_rows.extend(items[:n_calib])
        train_rows.extend(items[n_calib:])
    rng.shuffle(train_rows)
    rng.shuffle(calib_rows)
    if not calib_rows and train_rows:
        # tiny datasets: peel one row so temperature fit still runs
        calib_rows.append(train_rows.pop())
    return train_rows, calib_rows


def domain_adapt(
    train_records: List[Dict],
    calib_records: List[Dict],
    output_dir: Path | str,
    *,
    name: str = "OneSystem-domain",
    encoder: str = DEFAULT_ENCODER,
    epochs: int = 4,
    batch_size: int = 8,
    max_text_len: int = 512,
    max_label_len: int = 64,
    encoder_lr: float = 3e-5,
    head_lr: float = 1e-4,
    device: Optional[str] = None,
    seed: int = 42,
) -> Tuple["object", Dict]:
    """Fine-tune from ``encoder``, calibrate temperature, save under ``output_dir``.

    Returns ``(model, manifest)``. Short-utterance domains can keep
    ``max_text_len=512``; long documents should raise it (up to 8192 for GTE).
    """
    from transformers import AutoTokenizer

    from onesystem.evaluate import fit_temperature
    from onesystem.model import OneSystem
    from onesystem.modeling import OneSystemModel

    if not train_records:
        raise ValueError("train_records is empty")
    if not calib_records:
        raise ValueError("calib_records is empty — keep a held-out slice for temperature")

    output_dir = Path(output_dir)
    device = device or pick_device()
    seed_everything(seed)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    print(
        f"domain-adapt {name}: {len(train_records)} train / {len(calib_records)} calib "
        f"from {encoder} on {device}",
        flush=True,
    )
    network = OneSystemModel.from_encoder_name(
        encoder,
        name=name,
        version=__version__,
        max_text_len=max_text_len,
        max_label_len=max_label_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(encoder, trust_remote_code=True)
    network.to(device)

    cfg = TrainConfig(
        encoder=encoder,
        epochs=epochs,
        batch_size=batch_size,
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        max_text_len=max_text_len,
        max_label_len=max_label_len,
        output_dir=str(output_dir),
        skip_zero_shot=True,
        seed=seed,
    )
    pairs = pairs_from_records(train_records)
    run = train(network, tokenizer, pairs, cfg, device)

    model = OneSystem.from_parts(network, tokenizer, device)
    temperature, calib_nll = fit_temperature(model, calib_records)
    network.config.temperature = temperature
    model.clear_cache()
    calib_nll_out = None if calib_nll != calib_nll else round(float(calib_nll), 4)

    output_dir.mkdir(parents=True, exist_ok=True)
    network.save(output_dir, tokenizer=tokenizer)
    manifest = {
        "name": name,
        "version": __version__,
        "model_type": "onesystem-domain",
        "base_encoder": encoder,
        "temperature": temperature,
        "calib_nll": calib_nll_out,
        "train_rows": len(train_records),
        "calib_rows": len(calib_records),
        "train_run": run,
        "max_text_len": max_text_len,
        "max_label_len": max_label_len,
    }
    (output_dir / "onesystem.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"saved {name} → {output_dir} (T={temperature:.3f})", flush=True)
    return model, manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a domain-specific OneSystem checkpoint from labelled JSONL",
    )
    parser.add_argument("--train", required=True, help="training JSONL (OneSystem record format)")
    parser.add_argument("--calib", default=None, help="calibration JSONL; if omitted, split 10% from --train")
    parser.add_argument("--output", required=True, help="directory for the domain checkpoint")
    parser.add_argument("--name", default="OneSystem-domain", help="saved model name")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-text-len", type=int, default=512)
    parser.add_argument("--max-label-len", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-split", action="store_true", help="when auto-splitting, write train/calib next to --train")
    args = parser.parse_args(argv)

    train_path = Path(args.train)
    all_train = read_jsonl(train_path)
    if args.calib:
        train_records = all_train
        calib_records = read_jsonl(args.calib)
    else:
        train_records, calib_records = split_records_for_calib(all_train, seed=args.seed)
        print(f"auto-split: {len(train_records)} train / {len(calib_records)} calib", flush=True)
        if args.write_split:
            out_train = train_path.with_name(train_path.stem + "_fit.jsonl")
            out_calib = train_path.with_name(train_path.stem + "_calib.jsonl")
            write_jsonl(out_train, train_records)
            write_jsonl(out_calib, calib_records)
            print(f"wrote {out_train} and {out_calib}", flush=True)

    domain_adapt(
        train_records,
        calib_records,
        args.output,
        name=args.name,
        encoder=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_text_len=args.max_text_len,
        max_label_len=args.max_label_len,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
