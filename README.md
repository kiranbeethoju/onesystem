# OneSystem

OneSystem is a standalone **System 1 decision model**. You pass a document and the labels that are legal for a decision; it returns the chosen label, a calibrated confidence, and the full probability table in **one forward pass**. It does not generate text. Loading never pulls GLiNER or any other vendor decision checkpoint — the weights you download are OneSystem's own.

**Repo:** [github.com/kiranbeethoju/onesystem](https://github.com/kiranbeethoju/onesystem) · **Release:** [v0.2.0](https://github.com/kiranbeethoju/onesystem/releases/tag/v0.2.0)

---

## Install

Python 3.10+. CPU, CUDA, or Apple MPS.

```bash
git clone https://github.com/kiranbeethoju/onesystem.git
cd onesystem
pip install -e .
```

Google Colab:

```python
!git clone https://github.com/kiranbeethoju/onesystem.git
%cd onesystem
!pip install -q -e .

from onesystem.model import OneSystem
model = OneSystem.load(device="cuda")
```

`OneSystem.load()` uses `models/onesystem` when weights are present locally; otherwise it downloads the ~420 MB release into `~/.cache/onesystem` (override with `ONESYSTEM_HOME`). No Hugging Face token required.

---

## Decision types (what the model outputs)

OneSystem is **not** a chat model. Every call is a typed classification over a candidate set you define.

| Type | How you declare it | What you get back | When to use it |
|---|---|---|---|
| **Single-label (choice)** | `"task": ["a", "b", "c"]` | `label`, `confidence`, `probabilities` | Mutually exclusive classes: intent, triage, doc type |
| **Multi-label** | `"task": {"labels": [...], "multi_label": True, "threshold": 0.5}` | `labels` (list), `probabilities`, `threshold` | Several tags can fire: complaint areas, topics, symptoms |
| **Ordinal (score)** | Labels that look like `"0","1","2","3"` (or set `"ordinal": True`) | Same as single-label, plus `score` (expected level) | Urgency, severity, NPS-style scales |
| **Described labels** | `"labels": {"refund": "customer wants money back", ...}` | Same shapes; descriptions are encoded with the label | Short or ambiguous names (`yes`/`no`, codes) |
| **Abstention** | `classify(..., min_confidence=0.6)` | Weak heads return `label: None`, `abstain: True` | Route uncertain cases to a human |

**Input types**

| Field | Type | Notes |
|---|---|---|
| Text | `str` | One document / message / note per call (or a list via `classify_batch`) |
| Tasks | `dict` | Keys are task names; values are label lists or dict specs |
| Labels | `list[str]` or `dict[str, str]` | At least two candidates per task; gold labels must be in the set at train time |

**Output types (per task)**

```python
# single-label
{"label": "refund_request", "confidence": 0.96, "probabilities": {...}}

# ordinal
{"label": "2", "confidence": 0.57, "probabilities": {...}, "score": 1.63}

# multi-label
{"labels": ["billing", "mobile_app"], "probabilities": {...}, "threshold": 0.5}

# abstain
{"label": None, "confidence": 0.41, "probabilities": {...}, "abstain": True}
```

---

## How OneSystem is built

### Architecture

```
text  --"{task}: {text}"-->  encoder --> masked mean --> proj --> normalize --\
                                                                               cosine * exp(log_scale) / temperature --> softmax | sigmoid
label --"{task}: {label}"--> encoder --> masked mean --> proj --> normalize --/
```

1. **Encoder** — 110M BERT-style network, initialised from [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) (MIT), then fine-tuned. Weights are saved inside OneSystem; runtime never downloads BAAI again.
2. **Task conditioning** — task name is prefixed on both text and labels so the same word can mean different things under different heads.
3. **Shared projection** — starts as the identity, so zero-shot already uses the encoder's geometry; training sharpens it.
4. **Scoring** — learned-scale cosine; softmax for single-label, sigmoid for multi-label; logits divided by a **temperature** fit after training.
5. **Label cache** — for a fixed schema, label vectors are computed once; each request is one text encode per head.

### Training data and split

| | |
|---|---|
| Source | Public [`fastino/fast-decisions`](https://huggingface.co/datasets/fastino/fast-decisions) **development** split |
| Size | 1,700 rows, 17 domains, ~25 decision heads |
| Split | Per domain: **70% train** (1,190) · **10% calibration** (170) · **20% eval** (340) |
| Record shape | `{"input": str, "output": {"classifications": [{"task", "labels", "true_label", "multi_label"}]}}` |
| Recipe | 6 epochs, batch 16, AdamW (encoder 3e-5, head 1e-4), eager attention |
| Calibration | Temperature **2.1** fit on the calibration slice only |
| Holdout (v0.2.0) | Zero-shot **0.453** → fine-tuned **0.552** exact match (320/580 heads), ECE **0.074** |

Scores above are on a local holdout cut from the public development data — **not** Fastino's unpublished test benchmark.

### What ships

| Artifact | Where |
|---|---|
| `model.safetensors` (~420 MB) | [GitHub release v0.2.0](https://github.com/kiranbeethoju/onesystem/releases/tag/v0.2.0) |
| `config.json`, tokenizer, `onesystem.json`, `eval.json` | Repo + release |
| Code | Pure PyTorch + Transformers — no `gliner2`, no PEFT |

---

## Quick start

```python
from onesystem.model import OneSystem

model = OneSystem.load()

model.classify(
    "Stop the bot and get me a person.",
    {
        "handoff": ["yes", "no"],
        "sentiment": ["negative", "neutral", "positive"],
        "urgency": ["0", "1", "2", "3"],
        "areas": {"labels": ["billing", "mobile_app", "login"], "multi_label": True},
        "doc_type": {
            "labels": {
                "invoice": "a bill for goods or services",
                "contract": "a signed agreement",
            }
        },
    },
    min_confidence=0.6,
)
```

```bash
onesystem "Stop the bot and get me a person." --task handoff --labels yes,no
onesystem "Checkout crashed and I never got a receipt." --task areas --labels checkout,notifications,login --multi-label
```

---

## Cookbook

Recipes below are **schemas you define at inference time**. The model does not invent labels — you pass the legal set. For production, fine-tune or calibrate on your own domain data and keep `min_confidence` so weak cases go to a human.

Always start with:

```python
from onesystem.model import OneSystem
model = OneSystem.load()   # or device="cuda" / "mps" / "cpu"
```

### Clinical

**Triage acuity, specialty route, and red-flag tags from a short intake note.**

> Not a medical device. Do not use as the sole basis for diagnosis or treatment. Keep a clinician in the loop; use abstention on low confidence.

```python
note = """
52M with sudden crushing chest pain radiating to left arm for 40 minutes,
diaphoresis, history of hypertension. BP 168/102, HR 110. No fever.
"""

result = model.classify(
    note,
    {
        "acuity": ["0", "1", "2", "3", "4"],   # ordinal ESI-style scale → also returns "score"
        "disposition": {
            "labels": {
                "ed_now": "send to emergency department immediately",
                "urgent_clinic": "same-day clinic or urgent care",
                "routine": "non-urgent outpatient follow-up",
                "self_care": "home care with return precautions",
            }
        },
        "specialty": [
            "cardiology",
            "pulmonology",
            "gastroenterology",
            "neurology",
            "primary_care",
            "other",
        ],
        "red_flags": {
            "labels": [
                "chest_pain",
                "shortness_of_breath",
                "neuro_deficit",
                "severe_bleeding",
                "altered_mental_status",
            ],
            "multi_label": True,
            "threshold": 0.45,
        },
    },
    min_confidence=0.55,
)

# Example branch
if result["disposition"].get("abstain") or result["disposition"]["label"] == "ed_now":
    route = "page_on_call"
else:
    route = result["specialty"]["label"]
```

**Prior-auth / document type for a faxed clinical packet**

```python
result = model.classify(
    "Attached: progress note, MRI lumbar spine report, and failed PT summary for L4-L5.",
    {
        "doc_bundle": {
            "labels": ["progress_note", "imaging_report", "lab_panel", "pt_summary", "referral_letter"],
            "multi_label": True,
        },
        "auth_likelihood": {
            "labels": {
                "likely_complete": "packet has clinical justification for imaging auth",
                "needs_more_docs": "missing conservative-care or prior imaging evidence",
                "not_applicable": "not an authorization request",
            }
        },
    },
)
```

---

### Financial

**Support ticket → product intent, fraud risk, and urgency for a bank / fintech desk.**

```python
ticket = """
I see two $499 charges from ACME MARKET on my debit card this morning.
I only bought once. Please reverse the duplicate and freeze the card if needed.
"""

result = model.classify(
    ticket,
    {
        "intent": [
            "dispute_charge",
            "card_freeze",
            "balance_inquiry",
            "transfer_help",
            "other",
        ],
        "channel_urgency": ["0", "1", "2", "3"],
        "fraud_signal": {
            "labels": {
                "high": "likely unauthorized or duplicate fraud pattern",
                "medium": "needs agent review",
                "low": "ordinary customer request",
            }
        },
        "products": {
            "labels": ["debit_card", "credit_card", "checking", "wire", "mobile_app"],
            "multi_label": True,
            "threshold": 0.4,
        },
    },
    min_confidence=0.6,
)

if result["fraud_signal"]["label"] == "high" and not result["fraud_signal"].get("abstain"):
    queue = "fraud_ops"
elif result["intent"]["label"] == "dispute_charge":
    queue = "disputes"
else:
    queue = "general_support"
```

**KYC / document screening (classification only, not OCR)**

```python
result = model.classify(
    "Customer uploaded a passport bio page and a utility bill dated last month for address proof.",
    {
        "kyc_docs": {
            "labels": ["passport", "drivers_license", "utility_bill", "bank_statement", "selfie"],
            "multi_label": True,
        },
        "case_status": ["docs_complete", "docs_incomplete", "needs_manual_review"],
    },
)
```

---

### EdTech

**Student message → intent, subject, and escalation for a tutoring or LMS chatbot.**

```python
message = """
I submitted my calculus homework twice and the gradebook still shows zero.
Midterm is tomorrow — can a human check this?
"""

result = model.classify(
    message,
    {
        "intent": [
            "grade_issue",
            "content_question",
            "tech_support",
            "deadline_extension",
            "other",
        ],
        "subject": [
            "math",
            "science",
            "english",
            "history",
            "computer_science",
            "other",
        ],
        "needs_human": ["yes", "no"],
        "urgency": ["0", "1", "2", "3"],
        "topics": {
            "labels": ["grading", "lms_bug", "exam_prep", "account_access"],
            "multi_label": True,
        },
    },
    min_confidence=0.55,
)

if result["needs_human"]["label"] == "yes" or result["urgency"]["label"] in {"2", "3"}:
    action = "escalate_to_instructor"
else:
    action = f"bot_reply_{result['intent']['label']}"
```

**Assignment / content tagging for a course catalog**

```python
result = model.classify(
    "Week 4 lab: build a REST API in Python, deploy to a free tier, write a short design doc.",
    {
        "bloom_level": {
            "labels": {
                "remember": "recall facts",
                "understand": "explain concepts",
                "apply": "use a procedure in a new situation",
                "analyze": "break down relationships",
                "create": "produce an original artifact",
            }
        },
        "skills": {
            "labels": ["python", "apis", "devops", "writing", "databases"],
            "multi_label": True,
            "threshold": 0.4,
        },
        "difficulty": ["0", "1", "2", "3"],
    },
)
```

---

### E-commerce

**Order support → intent, policy path, and multi-label issue areas.**

```python
email = """
Order #18422 arrived with a cracked phone case. I need a replacement before Friday
or a refund to the original card. Tracking said delivered yesterday.
"""

result = model.classify(
    email,
    {
        "intent": [
            "refund_request",
            "replacement",
            "order_status",
            "cancel_order",
            "address_change",
            "other",
        ],
        "policy_path": {
            "labels": {
                "damaged_in_transit": "item arrived broken; open replacement or refund",
                "change_of_mind": "customer no longer wants the item",
                "late_delivery": "SLA miss; goodwill or carrier claim",
                "wrong_item": "fulfilled SKU does not match order",
            }
        },
        "urgency": ["0", "1", "2", "3"],
        "areas": {
            "labels": ["shipping", "product_quality", "payments", "account"],
            "multi_label": True,
            "threshold": 0.45,
        },
    },
    min_confidence=0.6,
)

# Cheap System-1 gate before a slower LLM reply
if result["intent"].get("abstain"):
    reply_mode = "human"
elif result["intent"]["label"] in {"refund_request", "replacement"}:
    reply_mode = "policy_macro"
else:
    reply_mode = "faq_bot"
```

**Catalog / review tagging**

```python
result = model.classify(
    "Love the fabric but the sizing runs small and checkout failed twice on Apple Pay.",
    {
        "sentiment": ["negative", "neutral", "positive"],
        "aspects": {
            "labels": ["sizing", "quality", "shipping", "checkout", "price"],
            "multi_label": True,
        },
        "nps_bucket": ["0", "1", "2", "3"],  # map your own scale
    },
)
```

---

## Pattern: System 1 gate in front of an LLM

Use OneSystem for the cheap, typed decision; call a generative model only when needed.

```python
def handle(text: str) -> str:
    d = model.classify(
        text,
        {
            "intent": ["refund_request", "order_status", "cancel_subscription", "other"],
            "handoff": ["yes", "no"],
        },
        min_confidence=0.65,
    )
    if d["handoff"]["label"] == "yes" or d["intent"].get("abstain"):
        return "queue:human"
    if d["intent"]["label"] == "order_status":
        return "macro:order_status"
    return "llm:compose_reply"   # only here
```

---

## What OneSystem brings

| | |
|---|---|
| Own weights | Full `model.safetensors`; no vendor decision checkpoint at load |
| Task-conditioned bi-encoder | Task prefix on text and labels; fixed schema ≈ one text pass per head |
| Typed outputs | Softmax / sigmoid / ordinal `score` / abstain |
| Calibrated confidence | Temperature on a held-out calib slice; ECE in the eval report |
| Label descriptions | Encode `{label: meaning}` when names are terse |
| Zero-shot row in the manifest | Untrained score on the same holdout before fine-tuning |
| Readable trainer | `python -m onesystem.train` — pure PyTorch |

---

## Train and evaluate

```bash
pip install -e ".[dev]"
python -m onesystem.train            # 6 epochs, batch 16; ~17 min on an M3 Pro
python -m onesystem.evaluate
python -m onesystem.evaluate --fit-temperature
```

Options: `--epochs`, `--batch-size`, `--max-text-len`, `--encoder`, `--output-dir`, `--skip-zero-shot`.

Published **v0.2.0** holdout: zero-shot 0.453 → **0.552** exact match (320/580 heads), ECE 0.074. Per-task breakdown: [`models/onesystem/README.md`](models/onesystem/README.md).

---

## Repository layout

| Path | Purpose |
|---|---|
| `onesystem/modeling.py` | Network, config, save/load |
| `onesystem/model.py` | `OneSystem.classify`, label cache, abstention |
| `onesystem/hub.py` | Local path or GitHub release download |
| `onesystem/data.py` | Download + train/calib/eval split |
| `onesystem/train.py` | Training + calibration + manifest |
| `onesystem/evaluate.py` | Exact match, ECE, temperature fit |
| `onesystem/cli.py` | `onesystem` CLI |
| `models/onesystem` | Config, tokenizer, manifest, eval (weights on the release) |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Useful: new labelled domains, harder negatives, multilingual runs, evaluations on untouched data.

## License

Apache 2.0 — [LICENSE](LICENSE), [NOTICE](NOTICE). Encoder init (`BAAI/bge-base-en-v1.5`, MIT) and training data (`fastino/fast-decisions`, Apache 2.0) keep their own licenses.

## Limitations

- Small specialist (~1.7k public development rows). Not a substitute for a large domain-tuned classifier or a clinical/financial decision system of record.
- Multi-label exact-set match remains the weakest metric.
- Calibrate and threshold on **your** labels before routing production traffic.
- English only in v0.2.0.
- Cookbook domains (clinical, finance, edtech, commerce) are **inference schemas**, not claims of medical or regulatory certification.
