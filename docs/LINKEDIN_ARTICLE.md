# LinkedIn article draft — OneSystem

Copy into LinkedIn → Write article. Suggested title below.

---

**Title:** OneSystem: a System 1 decision model you can actually gate production on

**Subtitle:** Typed labels, calibrated confidence, one forward pass — and cookbooks for support, commerce, fintech, edtech, and clinical schemas.

---

Most “AI for decisions” today means: call a chat model and hope the JSON comes back clean.

That’s System 2 — slow, expensive, and easy to overtrust. It’s great for drafting a reply. It’s a weak foundation for *routing*, *triage*, or *policy*.

I open-sourced **OneSystem**: a standalone System 1 decision model.

You pass text and the labels that are legal for a decision.  
It returns a typed answer — single choice, multi-label tags, or an ordinal score — plus calibrated confidence, in **one forward pass**.  
It does **not** generate text. Loading does **not** pull GLiNER or any other vendor decision checkpoint. The weights are OneSystem’s own (~420 MB on a GitHub release).

Repo: https://github.com/kiranbeethoju/onesystem  
Live cookbook (code + real outputs): https://kiranbeethoju.github.io/onesystem/

---

## The problem with “just ask the LLM”

Support, KYC queues, LMS bots, and intake forms need the same shape of answer every time:

- Which intent?
- Which tags?
- How urgent (0–3)?
- Should a human take over?

If you solve that only with a generative model, you pay latency and tokens on every turn, and you still need a parser, retries, and a confidence story you mostly invent.

System 1 should be cheap and typed. System 2 should compose language *after* the decision is made.

---

## How OneSystem works

1. You declare the legal labels for each decision head.  
2. Text and labels are encoded with a **task prefix** (`"{task}: {text}"` / `"{task}: {label}"`).  
3. Scores are cosine similarity with a learned scale, then softmax (single-label) or sigmoid (multi-label).  
4. A **temperature** fit on a calibration split turns logits into confidences you can threshold.  
5. Below `min_confidence` → explicit `abstain` → human.

Architecture (simplified):

```
text  → encoder → pool → proj → normalize ─┐
                                           cosine × scale / T → softmax | sigmoid
label → encoder → pool → proj → normalize ─┘
```

The encoder is a 110M BERT-style network initialised from `BAAI/bge-base-en-v1.5`, then fine-tuned. A shared projection starts as the identity so zero-shot already uses the encoder’s geometry. Training data is the public `fastino/fast-decisions` development split (train / calib / eval per domain). On our local holdout: **0.453 → 0.552** exact match after fine-tuning, ECE **0.074**, temperature **2.1**. That holdout is not Fastino’s unpublished test benchmark — say so when you cite the number.

**Context limits (v0.2.0):** text is truncated at **320 tokens** (~220–300 English words / ~1.4–2k characters, including the task prefix). Labels (name + optional description) cap at **32 tokens**. The underlying encoder can go to 512, but this release trains and infers at 320. Built for short tickets and notes — not full PDFs.

---

## Decision types (the data model)

| Type | You pass | You get |
|---|---|---|
| Single-label | `["refund", "other"]` | `label`, `confidence`, `probabilities` |
| Multi-label | `multi_label: True`, threshold | list of labels above threshold |
| Ordinal | `"0","1","2","3"` | same + expected `score` |
| Described labels | `{ "ed_now": "send to ED…" }` | same shapes; descriptions are encoded |
| Abstain | `min_confidence=0.6` | `label: null`, `abstain: true` |

---

## Examples across domains

Outputs below are from OneSystem **v0.2.0** (same as the GitHub Pages cookbook).

### 1) Support — handoff

**Input:** “Stop the bot and get me a person.”

**Heads:** handoff · sentiment · urgency  

**Output (abbrev.):** handoff = `yes` (0.98). Sentiment and urgency **abstain** under the floor — which is correct behaviour when the signal is weak.

### 2) E-commerce — damaged delivery

**Input:** cracked phone case, wants replacement or refund, tracking says delivered.

**Heads:** intent · policy_path (with descriptions) · urgency · multi-label areas  

**Output:** intent ≈ `replacement` (0.60), policy_path = `damaged_in_transit` (0.93), urgency abstains. That’s enough to open a replacement macro without waiting on an LLM.

### 3) Financial — duplicate debit charge

**Input:** two $499 charges, “reverse the duplicate and freeze the card if needed.”

**Heads:** intent · channel_urgency · fraud_signal · products (multi-label)  

**Output:** intent = `card_freeze` (0.72) — the freeze phrase wins over dispute. `fraud_signal` abstains. Don’t auto-close fraud; escalate.

### 4) EdTech — gradebook before midterm

**Input:** calculus homework submitted twice, gradebook still zero, midterm tomorrow, asks for a human.

**Heads:** intent · subject · needs_human · urgency · topics  

**Output:** intent = `grade_issue` (0.76), subject = `math` (0.80). Urgency abstains. Binary `yes`/`no` heads are brittle without descriptions — prefer described labels or rules that combine abstain + keywords.

### 5) Clinical schema demo — chest pain note

**Not a medical device.** Schema only.

**Heads:** acuity (ordinal) · disposition (described) · specialty · red_flags (multi-label)  

**Output:** disposition = `ed_now` (0.60), specialty = `cardiology` (0.61), acuity abstains. Abstention on acuity is a feature before a clinician — not a bug.

### 6) Pattern — System 1 in front of an LLM

```python
d = model.classify(text, {
    "intent": ["refund_request", "order_status", "cancel_subscription", "other"],
    "handoff": ["yes", "no"],
}, min_confidence=0.65)

if d["handoff"]["label"] == "yes" or d["intent"].get("abstain"):
    return "queue:human"
if d["intent"]["label"] == "order_status":
    return "macro:order_status"
return "llm:compose_reply"
```

On “billed twice… refund the duplicate”: intent = `refund_request` (0.96), handoff = `yes` (0.75) → **queue:human**.

---

## What I’m asking for

- Try the Colab / pip install path in the README.  
- Steal the cookbook schemas for your domain; fine-tune or recalibrate on your labels before production.  
- PRs welcome: new domains, multilingual runs, harder negatives, better evals on untouched data.

**Links**  
GitHub: https://github.com/kiranbeethoju/onesystem  
Release: https://github.com/kiranbeethoju/onesystem/releases/tag/v0.2.0  
Cookbook site: https://kiranbeethoju.github.io/onesystem/

#OpenSource #MachineLearning #System1 #NLP #MLOps #CustomerSupport #FinTech #EdTech #HealthTech #Ecommerce
