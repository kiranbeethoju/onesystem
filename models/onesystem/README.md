# OneSystem v0.2.0

Standalone System 1 decision model. The weights in this release are OneSystem's own; loading them does not pull any other vendor's decision checkpoint.

- Name: OneSystem
- Architecture: task-conditioned bi-encoder, 110M-parameter BERT-style encoder, masked-mean pooling, one shared projection, learned-scale cosine, scalar temperature
- Encoder initialisation: `BAAI/bge-base-en-v1.5` (MIT), before OneSystem training
- Data: `fastino/fast-decisions` development split, per domain 70 percent train (1,190 rows), 10 percent calibration (170), 20 percent eval (340)
- Recipe: 6 epochs, batch 16, AdamW (encoder 3e-5, head 1e-4), linear warmup and decay, eager attention
- Context: **320 tokens** text (incl. task prefix; ~220–300 English words), **32 tokens** per label; encoder hard max 512
- Calibration: temperature 2.1 fit on the calibration slice
- Local holdout: zero-shot 0.453 exact match before training, **0.552 exact match (320 of 580 heads) after training**, expected calibration error 0.074. This is not Fastino's unpublished test benchmark.
- License: Apache 2.0

## Files

| File | In git | In release |
|---|---|---|
| `config.json` | yes | yes |
| `model.safetensors` (about 440 MB) | no | yes |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt` | yes | yes |
| `onesystem.json` (training manifest) | yes | yes |
| `eval.json` (per-task holdout scores) | yes | yes |

`OneSystem.load()` reads this directory when `model.safetensors` is present, and otherwise downloads the release assets for the tag in `onesystem/__init__.py` into `~/.cache/onesystem`.

## Where it is strong and weak

Strong heads on the holdout: sentiment (0.82), sport, program, doc_type (0.80), format, feedback_type, asking_status (0.75). Weak heads: product_area (0.10), queue (0.20), urgency (0.25), and several yes/no heads near 0.5. Labels such as `yes` and `no` carry no meaning on their own, so a bi-encoder has to learn them from the task prefix and the roughly 70 training rows per head. Pass label descriptions (`{"labels": {"yes": "the message contains personal data", ...}}`) when you use such a head, and fit `min_confidence` on your own data.
