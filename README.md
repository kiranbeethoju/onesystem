# OneSystem

**A production-ready System‑1 decision engine for LLM pipelines.**

Pass a short document and the labels that are legal for a decision. OneSystem returns a typed answer, a probability table, and calibrated confidence in **one forward pass**. It does **not** generate text. It sits **in front of** chat / reasoning models so simple requests never pay for an LLM call.

**Repo:** [github.com/kiranbeethoju/onesystem](https://github.com/kiranbeethoju/onesystem) · **Release:** [v0.3.0](https://github.com/kiranbeethoju/onesystem/releases/tag/v0.3.0) · **Cookbook:** [kiranbeethoju.github.io/onesystem](https://kiranbeethoju.github.io/onesystem/) · **Benchmarks:** [BANKING77 + CLINC150](https://kiranbeethoju.github.io/onesystem/benchmarks.html)

---

## 1. What is OneSystem?

OneSystem is a **standalone classifier** (own weights, ~550 MB). Not a chat model. Not a LoRA on someone else’s decision checkpoint.

| You need | OneSystem |
|---|---|
| One label from a fixed set | Native (softmax) |
| Several tags at once | Native (sigmoid + threshold) |
| Urgency / severity scale | Ordinal `score` |
| Calibrated confidence | Temperature fit after training |
| “Don’t guess — escalate” | `min_confidence` → `abstain` |
| Draft an email / plan / tool calls | **No** — hand off to an LLM |

Architecturally closer to embedding classifiers (BGE / E5 / GTE) than to an LLM.

---

## 2. Why not use an LLM for every request?

Many tickets only need: document type? urgency? department? handoff?

That is **routing**, not reasoning. An LLM is slower, costlier, and less deterministic for fixed schemas.

| | OneSystem (System‑1) | Chat / reasoning LLM (System‑2) |
|---|---|---|
| Purpose | Typed decision engine | Reasoning, planning, generation |
| Generates text | No | Yes |
| Fixed schema + probabilities | Native | Prompted |
| Calibration & abstain | Designed in | Usually prompted |
| Latency / cost | Very low locally | Higher |
| Best for | Route, filter, prioritize | Synthesis, tools, drafts |

**Bottom line:** put OneSystem **in front of** your LLM — don’t replace the LLM with it.

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
Macro / workflow   LLM
 │                 │
Done             Reason · tools · draft
```

**Simple** — *“I need a refund for the duplicate March charge.”* → refund macro, no LLM.  
**Complex** — multi-step policy + draft email → OneSystem routes, then the LLM runs.

---

## 4. Thirty-second example

```bash
git clone https://github.com/kiranbeethoju/onesystem.git
cd onesystem && pip install -e .
```

```python
from onesystem.model import OneSystem

model = OneSystem.load()  # local weights or GitHub release (~550 MB)

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

Colab multi-use-case notebook: [`examples/colab_usecases.py`](examples/colab_usecases.py) (paste into Colab cells).

---

## 5. Performance, context, calibration

| Metric | v0.3.0 |
|---|---|
| Encoder init | [`Alibaba-NLP/gte-base-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) (Apache 2.0, **8192** ctx) |
| Text context | **8192 tokens** (~6k–7k English words), incl. task prefix |
| Label context | **128 tokens** |
| Local holdout (fast-decisions) | Zero-shot **0.457** → fine-tuned **0.590** · ECE **0.057** · T **2.5** |
| BANKING77 test (77-way) | Current **63.3%** → domain-adapted **94.0%** |
| CLINC150 in-scope (150-way) | Current **73.0%** → domain-adapted **97.6%**; OOS recall **91.7%** (threshold from val) |
| Weights | ~550 MB `model.safetensors` + `encoder/` architecture code |

Scores are dataset-specific — see [benchmarks](https://kiranbeethoju.github.io/onesystem/benchmarks.html). Prefer **2–16** labels per task in production schemas.

---

## 6. How it is trained

```
text  --"{task}: {text}"-->  encoder → mean → proj → normalize ─┐
                                                                cosine × scale / T → softmax | sigmoid
label --"{task}: {label}"--> encoder → mean → proj → normalize ─┘
```

- Data: public [`fastino/fast-decisions`](https://huggingface.co/datasets/fastino/fast-decisions) development split — 70% train / 10% calib / 20% eval per domain.
- Recipe: AdamW; **max_text_len=8192**, **max_label_len=128**; temperature on calib only.
- Pure PyTorch + Transformers — `python -m onesystem.train` / `onesystem.evaluate`.

---

## 7. Domain recipes

Schemas are yours at inference. Full examples + JSON: [kiranbeethoju.github.io/onesystem](https://kiranbeethoju.github.io/onesystem/).

**E-commerce** · **Financial** · **EdTech** · **Clinical (demo only, not a device)** · **Ops / incident urgency**

```python
model.classify(
    "The deploy failed twice and customers are seeing 500s. Can someone look now?",
    {
        "urgent": {
            "labels": {
                "yes": "Does this need attention right now?",
                "no": "This can wait",
            }
        }
    },
)
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
| `examples/colab_usecases.py` | Colab multi-use-case script |
| `docs/` | GitHub Pages cookbook |
| `models/onesystem` | Config, tokenizer, manifest (weights on the release) |

See [CONTRIBUTING.md](CONTRIBUTING.md). Apache 2.0 — [LICENSE](LICENSE), [NOTICE](NOTICE).
