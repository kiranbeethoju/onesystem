# OneSystem

LoRA adapter for operational classification. Load it with the base checkpoint `fastino/GLiNER2.5-Decide`.

- Name: OneSystem
- Base: `fastino/GLiNER2.5-Decide` (Apache 2.0)
- Data: `fastino/fast-decisions` development split, 1,360 train rows and 340 local holdout rows
- Method: one epoch, LoRA rank 8 on the encoder and classifier
- Local holdout: 0.679 exact match (394 of 580 heads, 340 rows). This is not Fastino's unpublished test benchmark.
- License: Apache 2.0

This directory is the saved adapter (`adapter_model.safetensors`), not a full copy of the base weights. See the repository README for training and evaluation.
