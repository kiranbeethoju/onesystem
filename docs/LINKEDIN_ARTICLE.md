# LinkedIn article draft — OneSystem

**Title:** OneSystem: fast typed decisions before expensive AI reasoning

**Subtitle:** A System‑1 engine in front of your LLM — not a replacement for it.

---

Most “AI for decisions” today means: call a chat model and hope the JSON comes back clean.

That’s System 2 — slow, expensive, and easy to overtrust.

I open-sourced **OneSystem**: a production-ready **System‑1 decision engine for LLM pipelines**.

You pass text and the labels that are legal for a decision.  
It returns a typed answer — single choice, multi-label tags, or an ordinal score — plus calibrated confidence, in **one forward pass**.  
It does **not** generate text. It sits **in front of** chat models so simple requests never pay for a reasoning call. Own weights (~550 MB). No third-party decision checkpoint at load.

**Repo:** https://github.com/kiranbeethoju/onesystem  
**Cookbook:** https://kiranbeethoju.github.io/onesystem/  
**Release v0.3.0:** 8192-token context (GTE-base backbone)

---

## How it works

1. Declare legal labels per decision head.  
2. Encode text and labels with a task prefix.  
3. Score with learned-scale cosine → softmax / sigmoid.  
4. Temperature calibration → confidences you can threshold.  
5. Below `min_confidence` → `abstain` → human or LLM.

**Context (v0.3.0):** 8192 tokens text · 128 tokens per label.  
**Holdout:** 0.457 → 0.590 exact match, ECE 0.057.

---

## Pattern

Simple refund → macro, no LLM.  
Complex multi-step + draft → OneSystem routes, LLM runs.

#OpenSource #MachineLearning #System1 #NLP #MLOps
