# OneSystem v0.3.0 (8k)

Standalone System 1 decision model with an **8192-token** context window.

- Name: OneSystem
- Encoder init: `Alibaba-NLP/gte-base-en-v1.5` (Apache 2.0, long-context embedding model)
- Context: **8192 tokens** text · **128 tokens** per label
- Architecture: task-conditioned bi-encoder, masked-mean pooling, shared identity-initialised projection, learned-scale cosine, temperature calibration
- Data: `fastino/fast-decisions` development split (70/10/20 train/calib/eval per domain)
- Recipe: 5 epochs, batch 8, AdamW (encoder 3e-5, head 1e-4)
- Calibration: temperature 2.5
- Local holdout: zero-shot **0.457** → fine-tuned **0.590** exact match (ECE 0.057). Not Fastino's unpublished test benchmark.
- License: Apache 2.0

`encoder/` holds the GTE architecture modules so `OneSystem.load()` does not need a Hub download for the backbone class. Weights live in `model.safetensors` (~550 MB).
