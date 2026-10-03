# OneSystem

**A production-ready System‑1 decision engine for LLM pipelines.**

Pass a short document and the labels that are legal for a decision. OneSystem returns a typed answer, a probability table, and calibrated confidence in **one forward pass**. It does **not** generate text. It does **not** replace reasoning models (JEV, Laya, GPT, …) — it sits **in front of them** so simple requests never pay for an LLM call.

**Repo:** [github.com/kiranbeethoju/onesystem](https://github.com/kiranbeethoju/onesystem) · **Release:** [v0.3.0](https://github.com/kiranbeethoju/onesystem/releases/tag/v0.3.0) · **Cookbook + live outputs:** [kiranbeethoju.github.io/onesystem](https://kiranbeethoju.github.io/onesystem/)

---

## 1. What is OneSystem?

OneSystem is a **standalone classifier** (own weights, ~420 MB), not a chat model and not a LoRA on someone else’s Decide checkpoint.

| You need | OneSystem |
|---|---|
| One label from a fixed set | Native (softmax) |
| Several tags at once | Native (sigmoid + threshold) |
| Urgency / severity scale | Ordinal `score` |
| Calibrated confidence | Temperature fit after training |
| “Don’t guess — escalate” | `min_confidence` → `abstain` |
| Draft an email / plan / tool calls | **No** — call JEV, Laya, or an LLM |

Architecturally it is closer to **BGE / E5 / ModernBERT classifiers / GLiNER‑Decide** than to an LLM.

---

## 2. Why not use an LLM for every request?

Medical note or support ticket often only needs:

* document type? · urgency? · department? · handoff? · appeal?

That is **routing**, not reasoning. A generative model is overkill: higher latency, higher cost, less deterministic JSON, and confidence you mostly invent with prompts.

| | OneSystem (System‑1) | JEV / Laya / LLM (System‑2) |
|---|---|---|
| Purpose | Typed decision engine | Reasoning, planning, generation |
| Generates text | No | Yes / sometimes |
| Fixed schema + probabilities | Native | Prompted / fragile |
| Calibration & abstain | Designed in | Usually prompted |
| Latency / cost | Very low / near zero locally | Higher / API or big GPU |
| Best for | Route, filter, prioritize | Policy synthesis, tools, drafts |

**Bottom line:** don’t replace JEV or Laya with OneSystem. **Put OneSystem in front of them.**

---

## 3. How it fits in a pipeline

```
Incoming request
        │
        ▼
    OneSystem          ← typed labels + confidence + abstain
        │
 ┌──────┴──────────┐
 │                 │
Simple           Complex / abstain
 │                 │
Macro / workflow   JEV · Laya · GPT
 │                 │
Done             Reason · tools · draft
```

**Simple** — *“I need a refund for the duplicate March charge.”*  
OneSystem: `intent=refund_request` (0.96), maybe `handoff=no` → run the refund macro. **No LLM.**

**Complex** — *“I ordered two phones, one is damaged, address changed, check policy and draft an email.”*  
OneSystem: high complexity / multi-intent / abstain → **call JEV or Laya** for planning and generation.

**Healthcare-shaped gate** (schema you define; not a medical device):

```
Clinical note → OneSystem → doc type · urgency · need human? · specialty · appeal?
                              │
                         only then → GPT / JEV / Laya
```

---

## 4. Thirty-second example

```bash
git clone https://github.com/kiranbeethoju/onesystem.git
cd onesystem && pip install -e .
```

```python
from onesystem.model import OneSystem

model = OneSystem.load()  # local weights or ~420 MB GitHub release

d = model.classify(
    "Hi, we were billed twice for March. Please refund the duplicate today.",
    {
        "intent": ["refund_request", "order_status", "cancel_subscription", "other"],
        "handoff": ["yes", "no"],
    },
    min_confidence=0.65,
)

if d["handoff"]["label"] == "yes" or d["intent"].get("abstain"):
    route = "human_or_llm"
else:
    route = f"macro:{d['intent']['label']}"
```

More code + JSON: [cookbook site](https://kiranbeethoju.github.io/onesystem/).

Colab: `pip install -e .` then `OneSystem.load(device="cuda")`. Set `ONESYSTEM_HOME` to move the cache.

---

## 5. Performance, context, calibration

| Metric | v0.3.0 |
|---|---|
| Encoder init | [`Alibaba-NLP/gte-base-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) (Apache 2.0, ~137M, **8192** ctx) |
| Text context | **8192 tokens** (~6k–7k English words / ~30–40k chars), incl. task prefix |
| Label context | **128 tokens** (name + optional description) |
| Local holdout | Zero-shot **0.457** → fine-tuned **0.590** exact match · ECE **0.057** · T **2.5** |
| Latency | Short tickets stay fast (dynamic pad); full 8k docs cost more |
| Weights | Full encoder in `model.safetensors` (~550 MB) + `encoder/` architecture code |

v0.2.x used BGE-base at **512** tokens. v0.3.0 uses a long-context **embedding** backbone (GTE) for 8k — plain LMs like ModernBERT collapsed cosine scores when fine-tuned for this task.

Holdout is a cut of the public development split — **not** Fastino’s unpublished test benchmark. Longer text is truncated (start kept). English. Prefer **2–16** labels per task.

**Output shapes:** single-label `{label, confidence, probabilities}` · multi-label `{labels, probabilities, threshold}` · ordinal adds `score` · weak heads can `{abstain: true, label: null}`.

---

## 6. How it is trained

```
text  --"{task}: {text}"-->  encoder → mean → proj → normalize ─┐
                                                                cosine × scale / T → softmax | sigmoid
label --"{task}: {label}"--> encoder → mean → proj → normalize ─┘
```

- Encoder initialised from [`Alibaba-NLP/gte-base-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) (Apache 2.0, 8192 ctx), then fine-tuned; saved inside OneSystem (no GLiNER at runtime).
- Data: public [`fastino/fast-decisions`](https://huggingface.co/datasets/fastino/fast-decisions) development split — 1,700 rows, 17 domains; per domain **70% train / 10% calib / 20% eval**.
- Recipe: 6 epochs, AdamW (encoder 3e-5, head 1e-4); **max_text_len=8192**, **max_label_len=128**; temperature fit on calib only.
- Pure PyTorch + Transformers — `python -m onesystem.train` / `onesystem.evaluate`.

```bash
pip install -e ".[dev]"
python -m onesystem.train
python -m onesystem.evaluate
```

---

## 7. Domain recipes (short)

Schemas are **yours at inference time**. Fine-tune / recalibrate on your labels before production. Full examples with **real JSON outputs**: [kiranbeethoju.github.io/onesystem](https://kiranbeethoju.github.io/onesystem/).

**E-commerce** — damaged order → `intent`, described `policy_path`, ordinal urgency, multi-label `areas`.  
**Financial** — duplicate charge → `dispute_charge` / `card_freeze`, `fraud_signal`, product tags.  
**EdTech** — gradebook issue → `grade_issue`, subject, `needs_human`, topic tags.  
**Clinical (demo only, not a device)** — intake note → disposition, specialty, red-flag multi-label; acuity can abstain → human.

```python
# Healthcare-shaped gate (illustrative)
note = "52M crushing chest pain to left arm 40 min, diaphoresis, HTN. BP 168/102."
r = model.classify(
    note,
    {
        "disposition": {
            "labels": {
                "ed_now": "send to emergency department immediately",
                "urgent_clinic": "same-day clinic or urgent care",
                "routine": "non-urgent outpatient follow-up",
                "self_care": "home care with return precautions",
            }
        },
        "specialty": ["cardiology", "pulmonology", "neurology", "primary_care", "other"],
        "need_human": ["yes", "no"],
    },
    min_confidence=0.55,
)
# if abstain or ed_now / need_human → clinician or LLM; else light workflow
```

CLI: `onesystem "Stop the bot and get me a person." --task handoff --labels yes,no`

---

## Layout, contribute, license

| Path | Role |
|---|---|
| `onesystem/model.py` | `classify` / abstain / label cache |
| `onesystem/modeling.py` | Network + save/load |
| `onesystem/hub.py` | Local dir or GitHub release download |
| `onesystem/train.py` / `evaluate.py` | Train, calibrate, ECE |
| `docs/` | GitHub Pages cookbook |
| `models/onesystem` | Config, tokenizer, manifest (weights on the release) |

See [CONTRIBUTING.md](CONTRIBUTING.md). Apache 2.0 — [LICENSE](LICENSE), [NOTICE](NOTICE). Upstream: BGE (MIT), fast-decisions (Apache 2.0).

**Not** a replacement for planning, tool use, or open-ended reasoning. **Is** a fast, calibrated filter/router before those systems.
