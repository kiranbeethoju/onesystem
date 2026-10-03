# OneSystem v0.2.1

Standalone System 1 decision model. The weights in this release are OneSystem's own; loading them does not pull any other vendor's decision checkpoint.

- Name: OneSystem
- Architecture: task-conditioned bi-encoder, 110M-parameter BERT-style encoder, masked-mean pooling, shared identity-initialised projection, learned-scale cosine, scalar temperature
- Encoder initialisation: `BAAI/bge-base-en-v1.5` (MIT), before OneSystem training
- **Context: 512 tokens text · 64 tokens per label** (encoder hard max is 512)
- Data: `fastino/fast-decisions` development split, per domain 70% train / 10% calibration / 20% eval
- Recipe: 6 epochs, batch 12, AdamW (encoder 3e-5, head 1e-4), eager attention
- Calibration: temperature 2.1
- Local holdout: zero-shot 0.453 → **0.543 exact match (315 of 580 heads)**, ECE 0.087. Not Fastino's unpublished test benchmark.
- License: Apache 2.0

To go beyond 512 tokens, retrain with a longer-context `--encoder`.
