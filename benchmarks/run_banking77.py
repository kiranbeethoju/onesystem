"""BANKING77: zero-shot v0.3.0 + domain-adapted fine-tune. Official test untouched."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from onesystem.benchmarks import (
    TASK,
    banking77_labels,
    evaluate_intent,
    read_banking77_csv,
    split_train_calib,
    to_onesystem_record,
    write_jsonl,
)
from onesystem.hub import LOCAL_MODEL_DIR

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmarks" / "banking77"
OUT = ROOT / "benchmarks" / "results"
ADAPTER_DIR = ROOT / "models" / "onesystem-banking77"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    from onesystem.evaluate import fit_temperature
    from onesystem.model import OneSystem
    from onesystem.train import TrainConfig, pairs_from_records, pick_device, train
    from onesystem.modeling import OneSystemModel
    from transformers import AutoTokenizer
    from onesystem import DEFAULT_ENCODER, __version__
    from onesystem.data import read_jsonl

    train_rows = read_banking77_csv(DATA / "train.csv")
    test_rows = read_banking77_csv(DATA / "test.csv")
    labels = banking77_labels(train_rows + test_rows)
    assert len(labels) == 77, len(labels)
    assert len(train_rows) == 10003, len(train_rows)
    assert len(test_rows) == 3080, len(test_rows)

    device = args.device or pick_device()
    report = {
        "dataset": "BANKING77",
        "license": "CC BY 4.0",
        "source": "https://github.com/PolyAI-LDN/task-specific-datasets",
        "task": TASK,
        "n_labels": 77,
        "label_wording": "raw category strings from the CSV (frozen)",
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "protocol": "Every test example sees all 77 candidate labels. Official test set never used for training or calibration.",
        "onesystem_version": __version__,
        "device": device,
    }

    OUT.mkdir(parents=True, exist_ok=True)

    if not args.skip_zero_shot:
        print(f"loading base OneSystem on {device}", flush=True)
        base = OneSystem.load(device=device)
        # Ensure we evaluate the published checkpoint config.
        report["base_checkpoint"] = {
            "source": str(LOCAL_MODEL_DIR if (LOCAL_MODEL_DIR / "model.safetensors").is_file() else "release"),
            "max_text_len": base.network.config.max_text_len,
            "temperature": base.network.config.temperature,
            "encoder_name": base.network.config.encoder_name,
        }
        print("BANKING77 zero-shot / current checkpoint …", flush=True)
        t0 = time.time()
        zs = evaluate_intent(base, test_rows, labels, task=TASK)
        zs["seconds"] = round(time.time() - t0, 1)
        report["current_checkpoint"] = zs
        print(json.dumps(zs, indent=2), flush=True)
        (OUT / "banking77_current.json").write_text(json.dumps({"banking77": report}, indent=2) + "\n")

    if not args.skip_finetune:
        train_part, calib_part = split_train_calib(train_rows, calib_ratio=0.1, seed=42)
        train_jsonl = DATA / "train_onesystem.jsonl"
        calib_jsonl = DATA / "calib_onesystem.jsonl"
        write_jsonl(train_jsonl, [to_onesystem_record(r["text"], r["category"], labels) for r in train_part])
        write_jsonl(calib_jsonl, [to_onesystem_record(r["text"], r["category"], labels) for r in calib_part])
        report["finetune_split"] = {
            "train_rows": len(train_part),
            "calib_rows": len(calib_part),
            "calib_ratio": 0.1,
            "seed": 42,
        }

        print(f"fine-tuning from {DEFAULT_ENCODER} on BANKING77 ({len(train_part)} train / {len(calib_part)} calib)", flush=True)
        network = OneSystemModel.from_encoder_name(
            DEFAULT_ENCODER,
            max_text_len=512,  # BANKING77 utterances are short; keep train cheap
            max_label_len=64,
            version=__version__,
        )
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_ENCODER, trust_remote_code=True)
        network.to(device)
        cfg = TrainConfig(
            encoder=DEFAULT_ENCODER,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_text_len=512,
            max_label_len=64,
            output_dir=str(ADAPTER_DIR),
            skip_zero_shot=True,
        )
        pairs = pairs_from_records(read_jsonl(train_jsonl))
        run = train(network, tokenizer, pairs, cfg, device)
        model = OneSystem.from_parts(network, tokenizer, device)
        temperature, calib_nll = fit_temperature(model, read_jsonl(calib_jsonl))
        network.config.temperature = temperature
        model.clear_cache()
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        network.save(ADAPTER_DIR, tokenizer=tokenizer)
        (ADAPTER_DIR / "onesystem.json").write_text(
            json.dumps(
                {
                    "name": "OneSystem-BANKING77",
                    "base": DEFAULT_ENCODER,
                    "dataset": "BANKING77",
                    "temperature": temperature,
                    "calib_nll": calib_nll,
                    "train_run": run,
                    "train_rows": len(train_part),
                    "calib_rows": len(calib_part),
                },
                indent=2,
            )
            + "\n"
        )

        print("BANKING77 domain-adapted eval on official test …", flush=True)
        t0 = time.time()
        ft = evaluate_intent(model, test_rows, labels, task=TASK)
        ft["seconds"] = round(time.time() - t0, 1)
        ft["temperature"] = temperature
        ft["calib_nll"] = round(calib_nll, 4)
        ft["train_run"] = run
        report["domain_adapted"] = ft
        print(json.dumps(ft, indent=2), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "banking77.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
