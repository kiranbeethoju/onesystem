# OneSystem — Google Colab multi-use-case demo
# Paste each cell into Colab (Runtime → GPU recommended).

# %% [markdown]
# # OneSystem Colab demo
# Typed System‑1 decisions (labels + confidence). No chat generation.
# Repo: https://github.com/kiranbeethoju/onesystem

# %% cell 1 — install
# !git clone https://github.com/kiranbeethoju/onesystem.git
# %cd onesystem
# !pip install -q -e .

# %% cell 2 — load
from onesystem.model import OneSystem
import json

model = OneSystem.load(device="cuda")  # use "cpu" if no GPU
print(model.name, "temp=", model.network.config.temperature,
      "max_text=", model.network.config.max_text_len)


def show(title, text, tasks, min_confidence=None):
    out = model.classify(text, tasks, min_confidence=min_confidence)
    print("\n" + "=" * 60)
    print(title)
    print("INPUT:", text)
    print("OUTPUT:")
    print(json.dumps(out, indent=2))
    return out


# %% cell 3 — ops / Noul-style urgency (yes/no)
show(
    "1) Ops — urgent?",
    "The deploy failed twice and customers are seeing 500s. Can someone look now?",
    {
        "urgent": {
            "labels": {
                "yes": "Does this need attention right now?",
                "no": "This can wait; no immediate attention needed",
            }
        }
    },
)

# %% cell 4 — e-commerce
show(
    "2) E-commerce — refund / replacement",
    "Order #18422 arrived with a cracked phone case. I need a replacement "
    "before Friday or a refund to the original card.",
    {
        "intent": [
            "refund_request", "replacement", "order_status",
            "cancel_order", "other",
        ],
        "policy_path": {
            "labels": {
                "damaged_in_transit": "item arrived broken",
                "change_of_mind": "customer no longer wants the item",
                "late_delivery": "delivery SLA miss",
                "wrong_item": "wrong SKU shipped",
            }
        },
        "urgency": ["0", "1", "2", "3"],
        "areas": {
            "labels": ["shipping", "product_quality", "payments", "account"],
            "multi_label": True,
            "threshold": 0.45,
        },
    },
    min_confidence=0.55,
)

# %% cell 5 — fintech
show(
    "3) Fintech — dispute / freeze",
    "I see two $499 charges from ACME MARKET on my debit card this morning. "
    "I only bought once. Please reverse the duplicate and freeze the card if needed.",
    {
        "intent": [
            "dispute_charge", "card_freeze", "balance_inquiry",
            "transfer_help", "other",
        ],
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

# %% cell 6 — edtech
show(
    "4) EdTech — gradebook issue",
    "I submitted my calculus homework twice and the gradebook still shows zero. "
    "Midterm is tomorrow — can a human check this?",
    {
        "intent": [
            "grade_issue", "content_question", "tech_support",
            "deadline_extension", "other",
        ],
        "subject": ["math", "science", "english", "history", "computer_science", "other"],
        "needs_human": ["yes", "no"],
        "urgency": ["0", "1", "2", "3"],
        "topics": {
            "labels": ["grading", "lms_bug", "exam_prep", "account_access"],
            "multi_label": True,
        },
    },
    min_confidence=0.55,
)

# %% cell 7 — clinical schema demo (NOT a medical device)
show(
    "5) Clinical schema demo (illustrative only)",
    "52M with sudden crushing chest pain radiating to left arm for 40 minutes, "
    "diaphoresis, history of hypertension. BP 168/102, HR 110. No fever.",
    {
        "disposition": {
            "labels": {
                "ed_now": "send to emergency department immediately",
                "urgent_clinic": "same-day clinic or urgent care",
                "routine": "non-urgent outpatient follow-up",
                "self_care": "home care with return precautions",
            }
        },
        "specialty": [
            "cardiology", "pulmonology", "gastroenterology",
            "neurology", "primary_care", "other",
        ],
        "red_flags": {
            "labels": [
                "chest_pain", "shortness_of_breath", "neuro_deficit",
                "severe_bleeding", "altered_mental_status",
            ],
            "multi_label": True,
            "threshold": 0.45,
        },
    },
    min_confidence=0.55,
)

# %% cell 8 — System-1 gate before an LLM
text = "Hi, we were billed twice for March. Please refund the duplicate today."
d = show(
    "6) Gate before LLM",
    text,
    {
        "intent": ["refund_request", "order_status", "cancel_subscription", "other"],
        "handoff": ["yes", "no"],
    },
    min_confidence=0.65,
)

if d["handoff"]["label"] == "yes" or d["intent"].get("abstain"):
    action = "queue:human_or_llm"
elif d["intent"]["label"] == "order_status":
    action = "macro:order_status"
else:
    action = "llm:compose_reply"
print("ROUTE →", action)

# %% cell 9 — batch
batch = model.classify_batch(
    [
        "Stop the bot and get me a person.",
        "Where is my package? Tracking has not moved in a week.",
        "Love the fabric but sizing runs small.",
    ],
    {
        "handoff": ["yes", "no"],
        "sentiment": ["negative", "neutral", "positive"],
    },
)
print("\n7) Batch")
print(json.dumps(batch, indent=2))

# %% [markdown]
# ## Domain-specific model
# Fine-tune when you have labelled data for a fixed taxonomy
# (see https://kiranbeethoju.github.io/onesystem/#domain-adapt).
#
# ```bash
# python -m onesystem.domain_adapt \
#   --train data/my_domain/train.jsonl \
#   --output models/onesystem-mydomain \
#   --name OneSystem-MyDomain --epochs 4
# ```
#
# ```python
# domain = OneSystem.load("models/onesystem-mydomain")  # or path on Colab
# domain.classify("…", {"intent": ["refund", "status", "other"]})
# ```
