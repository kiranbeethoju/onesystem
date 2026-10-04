# OneSystem — brief for another AI / ChatGPT

Copy everything below the line into ChatGPT (or any LLM) as project context.

---

## What this application is

**OneSystem** is an open-source **System‑1 decision / classification engine** for production LLM pipelines.

- **Repo:** https://github.com/kiranbeethoju/onesystem  
- **Author:** Kiran Beethoju  
- **License:** Apache 2.0  
- **Release:** v0.3.0 (~550 MB weights on GitHub Releases)  
- **Cookbook / demos:** https://kiranbeethoju.github.io/onesystem/  
- **Benchmarks:** https://kiranbeethoju.github.io/onesystem/benchmarks.html  

It is **not** a chatbot and **not** a text generator. You pass a document (ticket, chat turn, note) plus the **legal label set** for each decision. In **one forward pass** it returns typed outputs: chosen label(s), full probability table, calibrated confidence, optional ordinal score, and optional **abstain**.

**Product positioning:** *fast typed decisions before expensive AI reasoning* — a gate/router **in front of** chat or reasoning models. Simple requests run macros/workflows with **no LLM call**. Complex, low-confidence, or failed classifications escalate to a custom LLM API.

## Problem it solves

Many enterprise requests only need routing: intent? urgency? handoff? department?  
Using a full LLM for that is slow, costly, and non-deterministic. OneSystem is a local specialist classifier for fixed schemas; the LLM is reserved for drafting, multi-step policy, tools, and open-ended work.

## Core API

```python
from onesystem.model import OneSystem

model = OneSystem.load()  # general v0.3.0; device="cuda"|"mps"|"cpu"
# Domain checkpoints (after local fine-tune):
# OneSystem.load("models/onesystem-banking77")
# OneSystem.load("models/onesystem-clinc150")
# OneSystem.load("models/onesystem-hwu64")
# OneSystem.load("models/onesystem-goemotions")

result = model.classify(
    "Please refund the duplicate charge.",
    {
        "intent": ["refund_request", "order_status", "other"],  # single-label
        "areas": {"labels": ["billing", "shipping"], "multi_label": True},
        "urgency": ["0", "1", "2", "3"],  # ordinal → adds score
        "urgent": {"labels": {"yes": "needs attention now", "no": "can wait"}},
    },
    min_confidence=0.6,  # weak head → label=None, abstain=True
)
```

Also: `classify_batch`, CLI `onesystem`, train `python -m onesystem.train`, domain fine-tune `python -m onesystem.domain_adapt`, eval `python -m onesystem.evaluate`.

## Pipeline pattern (with LLM fallback)

```
Incoming request → OneSystem
  → Simple + confident → macro / workflow (Done, no LLM)
  → Complex / abstain / exception → call custom OpenAI-compatible LLM API
       (draft, reason, tools, plan)
```

Env for fallback: `LLM_API_URL`, `LLM_API_KEY`, `LLM_MODEL`. Same pattern works for OpenAI, Azure, vLLM, Groq, etc. Full code is in the cookbook § LLM fallback and README.

## Available model configs

| Role | Load path | Notes |
|---|---|---|
| General multi-domain | `OneSystem.load()` or `"models/onesystem"` | Release v0.3.0, 8192 ctx |
| BANKING77 | `"models/onesystem-banking77"` | After `benchmarks/run_banking77.py` → ~94% test |
| CLINC150 | `"models/onesystem-clinc150"` | ~97.6% in-scope; OOS via abstain |
| HWU64 | `"models/onesystem-hwu64"` | ~93.7% test |
| GoEmotions | `"models/onesystem-goemotions"` | Multi-label; ~61% macro-F1 |
| Custom domain | `"models/onesystem-mydomain"` | `python -m onesystem.domain_adapt --train …` |

Domain weights are trained locally (not all published as release assets). General weights download from GitHub release if missing.

## Architecture

- Task-conditioned **bi-encoder**: encode `"{task}: {text}"` and `"{task}: {label}"` (optional description).
- Masked-mean pool → **shared** identity-initialised projection → L2 normalize → cosine × learned scale → / temperature → softmax (single-label) or sigmoid (multi-label).
- Label embeddings **cached** for fixed schemas.
- Backbone init: **Alibaba-NLP/gte-base-en-v1.5** (8192). Full encoder saved inside OneSystem.
- Runtime does **not** load GLiNER, PEFT adapters, or any third-party decision checkpoint.
- Stack: PyTorch + Transformers (`>=4.49,<5`) + safetensors. Diagrams: `scripts/render_diagrams.py` → `assets/pipeline.png`, `assets/architecture.png`.

## Context limits (v0.3.0)

| Limit | Value |
|---|---|
| Text | 8192 tokens (incl. task prefix) |
| Label | 128 tokens |
| Language | English-focused |
| Typical use | tickets, notes, chat turns (full 8k only when needed) |

## Training & benchmarks

- Base training data: public `fastino/fast-decisions` **development** split (~1700 rows, 17 domains), 70/10/20 train/calib/eval per domain.
- Local holdout (not Fastino’s unpublished test): zero-shot **0.457** → fine-tuned **0.590** exact match, ECE **0.057**, T **2.5**.
- Public domain-adapted test scores (vs general checkpoint): BANKING77 **63.3% → 94.0%**; CLINC150 in-scope **73.0% → 97.6%**; HWU64 **64.5% → 93.7%**; GoEmotions macro-F1 **8.3% → 61.0%**.

## Package layout (local workspace often `systemOneRandD`)

- `onesystem/model.py` — public API  
- `onesystem/modeling.py` — network + save/load  
- `onesystem/hub.py` — local dir or GitHub release  
- `onesystem/domain_adapt.py` — fine-tune domain checkpoints  
- `onesystem/train.py` / `evaluate.py` — train, calibrate, ECE  
- `benchmarks/` — BANKING77, CLINC150, HWU64, GoEmotions  
- `docs/` — GitHub Pages cookbook  
- `examples/colab_usecases.py` — Colab demos  
- `models/onesystem*` — checkpoints (large weights gitignored / on release)

## What it is for

Enterprise **routing**: support intent, handoff, urgency, fraud tags, edtech escalation, ops “is this urgent?”, emotion tagging, clinical **schema demos only** (not a medical device). Cheap filter before calling an LLM for drafting/planning.

## What it is not

- Not a generative / chat model  
- Not a long-horizon planning or tool-calling agent by itself  
- Not a drop-in replacement for an LLM on open-ended tasks  
- Not certified for clinical/financial regulatory decisions out of the box  

## Install

```bash
git clone https://github.com/kiranbeethoju/onesystem.git
cd onesystem
pip install -e .
# from onesystem.model import OneSystem; OneSystem.load(device="cuda")
```

## Public comparison (raw vs Laya / OpenJev)

Board: https://kiranbeethoju.github.io/onesystem/comparison.html  

On official BANKING77 / CLINC150 / HWU64 tests (full label sets), **raw** OneSystem v0.3.0 outscored `pip install laya` and OpenJev (Verdict) in this harness. Fine-tuned track is OneSystem-only (competitors lack a documented `domain_adapt` equivalent here). OpenJev uses a ≤24-option tournament for large label sets — document that when citing.

When helping with this project: preserve Apache 2.0 attribution; do not claim Fastino unpublished test scores; keep OneSystem positioned as **complement to LLMs**, not a competitor that replaces them. Prefer “open domain-adaptable decision layer” over “new AI paradigm.”
