"""CLINC150: in-scope intent accuracy + OOS abstention (threshold from validation)."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

from onesystem.benchmarks import (
    TASK,
    choose_threshold,
    evaluate_intent,
    evaluate_oos,
    load_clinc150,
    split_train_calib,
    to_onesystem_record,
    write_jsonl,
)
from onesystem.hub import LOCAL_MODEL_DIR

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmarks" / "clinc150" / "data_full.json"
OUT = ROOT / "benchmarks" / "results"
ADAPTER_DIR = ROOT / "models" / "onesystem-clinc150"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    from transformers import AutoTokenizer

    from onesystem import DEFAULT_ENCODER, __version__
    from onesystem.data import read_jsonl
    from onesystem.evaluate import fit_temperature
    from onesystem.model import OneSystem
    from onesystem.modeling import OneSystemModel
    from onesystem.train import TrainConfig, pairs_from_records, pick_device, train

    data = load_clinc150(DATA)
    labels = data["labels"]
    assert len(labels) == 150, len(labels)

    device = args.device or pick_device()
    report = {
        "dataset": "CLINC150",
        "source": "https://github.com/clinc/oos-eval",
        "task": TASK,
        "n_labels": 150,
        "label_wording": "raw intent strings from data_full.json (frozen)",
        "splits": {k: len(v) for k, v in data.items() if k != "labels"},
        "protocol": (
            "In-scope classification uses all 150 intents. "
            "Abstention threshold chosen on val + oos_val only; test + oos_test held out."
        ),
        "onesystem_version": __version__,
        "device": device,
    }
    OUT.mkdir(parents=True, exist_ok=True)

    def full_eval(model, tag: str) -> Dict:
        print(f"{tag}: in-scope test classification …", flush=True)
        t0 = time.time()
        in_metrics = evaluate_intent(model, data["test"], labels, task=TASK)
        in_metrics["seconds"] = round(time.time() - t0, 1)
        print(f"{tag}: choosing threshold on val+oos_val …", flush=True)
        threshold, grid = choose_threshold(model, data["val"], data["oos_val"], labels, task=TASK)
        print(f"{tag}: threshold={threshold}", flush=True)
        print(f"{tag}: OOS / abstention on test …", flush=True)
        oos_metrics = evaluate_oos(
            model, data["test"], data["oos_test"], labels, min_confidence=threshold, task=TASK
        )
        # Also accepted metrics with that threshold on in-scope test
        with_floor = evaluate_intent(
            model, data["test"], labels, task=TASK, min_confidence=threshold, measure_latency=False
        )
        return {
            "in_scope_forced": in_metrics,
            "threshold_selection": {"chosen": threshold, "validation_grid_summary": [
                {k: g[k] for k in ("min_confidence", "oos_recall", "in_scope_rejection_rate", "coverage", "accepted_accuracy")}
                for g in grid["grid"]
            ]},
            "test_with_abstention": oos_metrics,
            "in_scope_with_threshold": {
                "coverage": with_floor["coverage"],
                "accepted_accuracy": with_floor["accepted_accuracy"],
                "abstain_rate": with_floor["abstain_rate"],
            },
        }

    if not args.skip_zero_shot:
        print(f"loading base OneSystem on {device}", flush=True)
        base = OneSystem.load(device=device)
        report["base_checkpoint"] = {
            "source": str(LOCAL_MODEL_DIR if (LOCAL_MODEL_DIR / "model.safetensors").is_file() else "release"),
            "max_text_len": base.network.config.max_text_len,
            "temperature": base.network.config.temperature,
            "encoder_name": base.network.config.encoder_name,
        }
        report["current_checkpoint"] = full_eval(base, "current")
        (OUT / "clinc150_current.json").write_text(json.dumps({"clinc150": report}, indent=2) + "\n")

    if not args.skip_finetune:
        train_part, calib_part = split_train_calib(data["train"], calib_ratio=0.1, seed=42)
        train_jsonl = DATA.parent / "train_onesystem.jsonl"
        calib_jsonl = DATA.parent / "calib_onesystem.jsonl"
        write_jsonl(train_jsonl, [to_onesystem_record(r["text"], r["category"], labels) for r in train_part])
        write_jsonl(calib_jsonl, [to_onesystem_record(r["text"], r["category"], labels) for r in calib_part])
        report["finetune_split"] = {"train_rows": len(train_part), "calib_rows": len(calib_part)}

        print(f"fine-tuning CLINC150 ({len(train_part)} train) …", flush=True)
        network = OneSystemModel.from_encoder_name(
            DEFAULT_ENCODER, max_text_len=512, max_label_len=64, version=__version__
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
        run = train(network, tokenizer, pairs_from_records(read_jsonl(train_jsonl)), cfg, device)
        model = OneSystem.from_parts(network, tokenizer, device)
        temperature, calib_nll = fit_temperature(model, read_jsonl(calib_jsonl))
        network.config.temperature = temperature
        model.clear_cache()
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        network.save(ADAPTER_DIR, tokenizer=tokenizer)
        adapted = full_eval(model, "domain_adapted")
        adapted["temperature"] = temperature
        adapted["calib_nll"] = round(calib_nll, 4)
        adapted["train_run"] = run
        report["domain_adapted"] = adapted

    path = OUT / "clinc150.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
