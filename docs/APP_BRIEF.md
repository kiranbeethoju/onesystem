# OneSystem — brief for ChatGPT / Gemini

Copy everything below the line into ChatGPT (or Gemini) as project context.

---

# OneSystem — open System‑1 decision engine (vs Jev / OpenJev / Laya)

## One-liner

**OneSystem** is an open-source, self-hostable **System‑1 decision layer** for LLM pipelines. You pass a short document + the legal label set; it returns a **typed decision, full probability table, calibrated confidence, and optional abstain** in one forward pass — **no generated text**. It sits **in front of** chat/reasoning models so simple routing never pays for an LLM call. When confidence is low or the case is complex, it **hands off to your LLM API**.

- **Repo:** https://github.com/kiranbeethoju/onesystem  
- **Author:** Kiran Beethoju · **License:** Apache 2.0 · **Release:** v0.3.0 (~550 MB)  
- **Cookbook:** https://kiranbeethoju.github.io/onesystem/  
- **LLM fallback docs:** https://kiranbeethoju.github.io/onesystem/#llm-fallback  
- **Benchmarks:** https://kiranbeethoju.github.io/onesystem/benchmarks.html  
- **Comparison board:** https://kiranbeethoju.github.io/onesystem/comparison.html  
- **This brief (in repo):** docs/APP_BRIEF.md  

## Same job as Jev — different product shape

Jev AI (TypeSafe) popularized: *state + structured question → typed decision + probabilities* (Choice / Noul / Score). OneSystem targets that same **System‑1 slot**, but as **own weights you run locally**, with **domain fine-tuning**, **multi-label**, **abstain**, and a **production LLM fallback** pattern.

| Attribute | Jev AI (hosted) | OpenJev / Verdict | Laya | **OneSystem** |
|---|---|---|---|---|
| Role | Hosted System‑1 API | Open decision model | Open classifier-style model | Open System‑1 decision engine |
| Output | Typed + probs | Typed + probs | Label + scores | Typed + probs + **abstain** |
| Shapes | Choice / Noul / Score | Choice-style (API-limited) | Classification-oriented | Single-label, **multi-label**, ordinal score, yes/no |
| Runs | Hosted (early access) | Local / open weights | Local | Local (CPU / MPS / CUDA) |
| Train on your taxonomy | Closed | Limited in public harness | Limited | **`domain_adapt`** |
| Escalation | Threshold → human | DIY | DIY | **`min_confidence` → abstain → your LLM API** |
| Context | Product-specific | Smaller open models | Shorter on large sets | **8192-token** text |
| Max options (practical) | Up to 255 (hosted claim) | Often **≤24** → tournament | Full set, slower | Full softmax / sigmoid |
| Lock-in | Commercial / waitlist | Open weights vary | Open | **Apache 2.0**, no API key for inference |

**Fair note:** Hosted TypeSafe Jev was **not** evaluated. Public board compares OneSystem vs `pip install laya` and Verdict-open-jev (usable open OpenJev; PyPI `openjev==0.0.1` is an empty stub).

## Head-to-head (raw / zero-shot, official tests)

| Dataset | OneSystem | OpenJev | Laya |
|---|---|---|---|
| BANKING77 (77-way) | **63.3%** | 53.3% | 38.0% |
| CLINC150 (150-way) | **73.0%** | 55.2% | 53.2% |
| HWU64 (64-way) | **64.5%** | 29.7% | 35.2% |

**Fine-tuned OneSystem only** (domain_adapt on that dataset’s train split): BANKING77 **94.0%**, CLINC150 **97.6%**, HWU64 **93.7%**. GoEmotions multi-label: macro-F1 **61.0%**. Local latency in harness: OneSystem ~20–40 ms p50.

---

## LLM calling — the production pattern (strong differentiator)

Jev-style models stop at “typed decision + probability.” OneSystem is designed as the **gate in front of an LLM**:

```
Incoming request
    → OneSystem.classify(..., min_confidence=0.65)
        → Simple + confident  → run a macro / workflow   (NO LLM call, cheap & fast)
        → abstain / complex / error → call your LLM API  (draft, reason, tools, plan)
```

### When OneSystem calls the LLM

| Trigger | What happens |
|---|---|
| `abstain=True` (confidence below `min_confidence`) | Escalate — don’t guess |
| Label is `other` / `complex` / `None` | Treat as “needs reasoning” |
| Exception during classify | Fail open to LLM with reason |
| Explicit handoff head (`handoff=yes`) | Route to human or LLM chat |

### Env vars (OpenAI-compatible — works with OpenAI, Azure, vLLM, Groq, local servers)

- `LLM_API_URL` — default `https://api.openai.com/v1/chat/completions`  
- `LLM_API_KEY` — required for live calls  
- `LLM_MODEL` — default `gpt-4o-mini` (or whatever your endpoint serves)  

### Reference handler (paste-ready)

```python
import json, os, urllib.request
from onesystem.model import OneSystem

model = OneSystem.load()  # or a domain checkpoint
LLM_URL = os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
LLM_KEY = os.environ["LLM_API_KEY"]

def call_llm(text: str, reason: str) -> str:
    body = {
        "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": f"OneSystem could not resolve this ({reason}). Help the user."},
            {"role": "user", "content": text},
        ],
    }
    req = urllib.request.Request(
        LLM_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())["choices"][0]["message"]["content"]

def handle(text, labels):
    try:
        intent = model.classify(text, {"intent": labels}, min_confidence=0.65)["intent"]
    except Exception as exc:
        return {"route": "llm", "reply": call_llm(text, f"error:{exc}")}
    if intent.get("abstain") or intent.get("label") in {None, "other", "complex"}:
        return {"route": "llm", "reply": call_llm(text, "abstain_or_complex"), "decision": intent}
    return {"route": "macro", "action": f"macro:{intent['label']}", "decision": intent}
```

**Why this matters vs Jev / OpenJev / Laya:** those products give you a decision. OneSystem gives you a **decision + a clear contract for when to spend LLM tokens**. Most tickets never hit the LLM; hard ones get full reasoning/generation. That is the cost/latency win.

Cookbook: https://kiranbeethoju.github.io/onesystem/#llm-fallback · Colab: `examples/colab_usecases.py`

---

## Other strong features (full stack)

### 1) Multi-head decisions in one forward pass
One `classify(...)` can return several typed answers at once: intent (single-label), tags (multi-label), urgency (ordinal score), handoff (yes/no).

### 2) Multi-label natively
Sigmoid + threshold — several tags can fire together (e.g. GoEmotions). Not limited to “pick exactly one.”

### 3) Ordinal Score (Jev-Score equivalent)
Pass ordered levels `["0","1","2","3"]` → per-level probabilities + probability-weighted `score` you can threshold.

### 4) Abstain / calibrated confidence
Temperature calibration after training; `min_confidence` → `label=None`, `abstain=True`. ECE reported in eval. OOS (out-of-scope) can be handled via abstain thresholds (CLINC150 OOS recall ~91.7% with threshold from val).

### 5) Domain adapt on *your* taxonomy
```bash
python -m onesystem.domain_adapt \
  --train data/my_domain/train.jsonl \
  --output models/onesystem-mydomain \
  --name OneSystem-MyDomain \
  --epochs 4
```
Then `OneSystem.load("models/onesystem-mydomain")`. Public FT: banking/clinc/hwu ~94–98%.

### 6) Batch inference
`classify_batch(texts, tasks)` for map-reduce style classification over large datasets (same schema, cached label embeddings).

### 7) Long context
**8192 tokens** text (incl. task prefix), **128 tokens** per label — real tickets/notes, not only playground one-liners. Backbone: Alibaba-NLP/gte-base-en-v1.5 (Apache 2.0).

### 8) Label embedding cache
Fixed schemas → label vectors computed once and reused → fast repeated decisions.

### 9) Self-host, offline, no waitlist
~550 MB `model.safetensors` + encoder; GitHub release download via `OneSystem.load()`; CPU / Apple MPS / CUDA. No hosted API key for inference.

### 10) Train / evaluate / calibrate CLI
`python -m onesystem.train` · `onesystem.evaluate` · `--fit-temperature` · ECE / NLL metrics. Pure PyTorch + Transformers (`>=4.49,<5`).

### 11) CLI for quick checks
`onesystem "Stop the bot and get me a person." --task handoff --labels yes,no`

### 12) Open cookbook + domain recipes
E-commerce, financial, edtech, ops/incident urgency, emotion tagging; clinical **schema demos only** (not a medical device). Diagrams: pipeline + architecture PNGs.

### 13) Transparent architecture (not a chat LLM)
Task-conditioned **bi-encoder**: encode `"{task}: {text}"` and `"{task}: {label}"` → masked-mean pool → shared projection → L2 normalize → cosine × learned scale → / temperature → softmax or sigmoid. Runtime does **not** load GLiNER, PEFT, or third-party decision checkpoints.

### 14) Proven on public intent benchmarks
Raw and domain-adapted numbers published; fair board vs Laya / OpenJev on BANKING77 / CLINC150 / HWU64.

---

## Core API

```python
from onesystem.model import OneSystem

model = OneSystem.load()  # general v0.3.0
# Domain: OneSystem.load("models/onesystem-banking77")  # after domain_adapt / benchmark train

result = model.classify(
    "Please refund the duplicate charge.",
    {
        "intent": ["refund_request", "order_status", "other"],
        "areas": {"labels": ["billing", "shipping"], "multi_label": True},
        "urgency": ["0", "1", "2", "3"],  # ordinal → adds score
        "urgent": {"labels": {"yes": "needs attention now", "no": "can wait"}},
    },
    min_confidence=0.6,
)
# → typed labels, full probs, score, abstain flags per head
```

Prefer **2–16** labels per production task head (large sets work; accuracy is dataset-specific).

---

## What OneSystem can do **beyond** Jev / OpenJev / Laya (summary)

1. **LLM fallback contract** — abstain/complex/error → OpenAI-compatible chat API; easy cases skip the LLM  
2. **Domain adapt** to your label set with published 94–98% domain scores  
3. **Multi-label + multi-head** in one call  
4. **Ordinal score + yes/no + choice** together  
5. **Abstain** as a first-class product feature  
6. **Self-host Apache 2.0** — no TypeSafe waitlist / no API for inference  
7. **8192-token** context for real documents  
8. **Batch classify** for dataset-scale routing  
9. **Calibration + ECE** tooling built in  
10. **Beat open Laya & OpenJev** on the public raw board (hosted Jev not measured)  

---

## Mental model

```
Traditional LLM:  Input → Prompt → Generate → Parse → Validate → Decision
Jev / OneSystem:  State + structured question → Typed decision + probabilities

OneSystem twist:  Decision → if abstain/complex → YOUR LLM API (draft / reason / tools)
                  else → macro / workflow (zero LLM cost)
```

OneSystem is **not** a chatbot and does **not** replace an LLM. Pattern: **LLM plans and writes; OneSystem makes the bounded calls** (route, urgency, tool pick, stop/continue) — and **skips the LLM** when confidence is high.

## What it is for / not for

**For:** support intent routing, handoff, urgency/severity, agent tool selection, spam/safety filters, lead scoring, map-reduce classification, emotion tags, ops “is this urgent?”, cheap gate before LLM drafting.

**Not for:** open-ended chat, long-horizon planning, writing emails, inventing new labels, certified clinical/financial regulatory decisions out of the box.

## How assistants should talk about it

- Complement to LLMs, not a replacement — emphasize the **LLM fallback gate**  
- Prefer “**open, domain-adaptable decision layer with LLM handoff**”  
- Cite comparison.html for Laya/OpenJev; do **not** claim hosted TypeSafe Jev was beaten without measuring it  
- Do not claim Fastino unpublished test scores  
- Preserve Apache 2.0 attribution (Kiran Beethoju / onesystem)

## Install

```bash
git clone https://github.com/kiranbeethoju/onesystem.git
cd onesystem && pip install -e .
# from onesystem.model import OneSystem; OneSystem.load(device="cuda")  # or mps / cpu
```
