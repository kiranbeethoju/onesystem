# OneSystem — brief for another AI / ChatGPT

Copy everything below into ChatGPT as project context.

---

## What this app is

**OneSystem** is an open-source **System-1 decision / classification model** (Python package + trained weights).  

- **Repo:** https://github.com/kiranbeethoju/onesystem  
- **Author:** Kiran Beethoju  
- **License:** Apache 2.0  
- **Current release:** v0.3.0  
- **Site:** https://kiranbeethoju.github.io/onesystem/

It is **not** a chatbot and **not** a text generator. You give it a short document (or up to ~8k tokens) plus the **legal label set** for each decision. It returns typed outputs: chosen label(s), full probability table, calibrated confidence, optional ordinal score, optional abstain.

Positioning: **fast typed decisions before expensive AI reasoning** — a gate/router in front of LLMs in production pipelines.

## Core API

```python
from onesystem.model import OneSystem
model = OneSystem.load()  # or device="cuda" / "mps" / "cpu"
result = model.classify(
    "text here",
    {
        "intent": ["refund_request", "order_status", "other"],           # single-label
        "areas": {"labels": ["billing", "login"], "multi_label": True}, # multi-label
        "urgency": ["0", "1", "2", "3"],                                 # ordinal → + score
        "urgent": {"labels": {"yes": "needs attention now", "no": "can wait"}},
    },
    min_confidence=0.6,  # weak heads → label=None, abstain=True
)
```

Also: `classify_batch`, CLI `onesystem`, train `python -m onesystem.train`, eval `python -m onesystem.evaluate`.

## Architecture

- Task-conditioned **bi-encoder**: encode `"{task}: {text}"` and `"{task}: {label}"` (optional description).
- Masked-mean pool → **shared** linear projection (identity-initialised) → L2 normalize → cosine × learned scale → / temperature → softmax or sigmoid.
- Label embeddings **cached** for fixed schemas (one text encode per task after warmup).
- v0.3.0 backbone init: **Alibaba-NLP/gte-base-en-v1.5** (8192 context). Full encoder weights saved inside OneSystem; runtime does not download GLiNER or any other decision vendor checkpoint.
- Custom encoder code for GTE lives under `models/onesystem/encoder/` (also release assets `encoder-*.py/json`).

## Context limits (v0.3.0)

| Limit | Value |
|---|---|
| Text | 8192 tokens (incl. task prefix) |
| Label | 128 tokens |
| Language | English |
| Typical use | tickets, notes, messages (not required to use full 8k) |

## Training data & metrics

- Public dataset: `fastino/fast-decisions` **development** split (~1700 rows, 17 domains).
- Split per domain: 70% train / 10% calibration / 20% eval.
- v0.3.0 holdout: zero-shot exact match **0.457** → fine-tuned **0.590**, ECE **0.057**, temperature **2.5**.
- Scores are **not** Fastino’s unpublished official test benchmark.

## Distribution

- Weights on GitHub Releases (`model.safetensors` ~550 MB), not in git.
- `OneSystem.load()` uses local `models/onesystem` if present, else downloads release `v0.3.0` to `~/.cache/onesystem`.
- Stack: PyTorch, Transformers, safetensors, huggingface_hub. No gliner2/peft at runtime.

## Package layout

- `onesystem/modeling.py` — network, config, save/load  
- `onesystem/model.py` — public `OneSystem` API  
- `onesystem/hub.py` — resolve local vs GitHub release  
- `onesystem/data.py` — download + splits  
- `onesystem/train.py` / `evaluate.py` — train + calibrate + ECE  
- `onesystem/cli.py` — CLI  
- `examples/colab_usecases.py` — Colab multi-domain demos  
- `docs/` — GitHub Pages  

## Decision types

1. Single-label → `{label, confidence, probabilities}`  
2. Multi-label → `{labels, probabilities, threshold}`  
3. Ordinal numeric labels → adds `score` (expected level)  
4. Described labels → `{label: description}` encoded with the label  
5. Abstain → `min_confidence` → `{label: null, abstain: true}`  

## What it is for

Enterprise **routing**: support intent, handoff, urgency, fraud tags, edtech escalation, ops “is this urgent?”, clinical **schema demos** (not a medical device). Cheap filter before calling an LLM for drafting/planning.

## What it is not

- Not a generative model  
- Not long-horizon planning / tool-calling agent  
- Not a replacement for an LLM on open-ended tasks  
- Not certified for clinical/financial regulatory decisions out of the box  

## Install (Colab / local)

```bash
git clone https://github.com/kiranbeethoju/onesystem.git
cd onesystem
pip install -e .
# then: from onesystem.model import OneSystem; OneSystem.load(device="cuda")
```
