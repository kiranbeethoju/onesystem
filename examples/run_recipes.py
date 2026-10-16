#!/usr/bin/env python3
"""Run cookbook domain recipes offline (local weights, no network).

Usage (from repo root, with deps already installed or PYTHONPATH set):

    python examples/run_recipes.py
    python examples/run_recipes.py --model models/onesystem --out docs/data/recipes.json

Clinical / medical schemas are illustrative only — not a medical device.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def polarity(finding: str) -> dict:
    """Per-finding polarity head — preferred over one global negation label.

    Keep descriptions short and contrastive; long prose dilutes the bi-encoder.
    """
    return {
        "labels": {
            "affirmed": f"{finding}: present / positive",
            "negated": f"{finding}: denied / no / without",
            "uncertain": f"{finding}: possible / suspected / cannot exclude",
            "not_mentioned": f"{finding}: not in this note",
        }
    }


CASES = [
    {
        "id": "medical_negated_dx",
        "domain": "clinical_nlp",
        "title": "Negated findings + incomplete history",
        "text": (
            "Patient denies chest pain and denies history of myocardial infarction. "
            "Past medical history is not available at this time. No known drug allergies stated."
        ),
        "tasks": {
            # Ask one polarity question per concept you care about
            "chest_pain": polarity("chest pain"),
            "mi_history": polarity("history of myocardial infarction / heart attack"),
            "pmh_availability": {
                "labels": {
                    "documented": "PMH listed with conditions",
                    "not_available": "PMH unavailable / unknown / not obtained",
                    "none_significant": "no significant past medical history",
                }
            },
            "history_quality": {
                "labels": {
                    "usable": "enough history to triage",
                    "need_more": "must ask for missing history",
                }
            },
            "next_action": {
                "labels": {
                    "extract_ok": "fields clear enough to store",
                    "ask_history": "ask for missing past medical history",
                    "confirm_negation": "confirm a denial is real",
                    "escalate_llm": "hand off to LLM or clinician",
                }
            },
        },
        "min_confidence": 0.4,
        "focus": "per-finding negation + PMH availability",
    },
    {
        "id": "medical_affirmed_with_hx",
        "domain": "clinical_nlp",
        "title": "Affirmed complaint with usable history",
        "text": (
            "52M with sudden crushing chest pain radiating to left arm for 40 minutes, "
            "diaphoresis. History of hypertension and type 2 diabetes. BP 168/102, HR 110. No fever."
        ),
        "tasks": {
            "chest_pain": polarity("chest pain"),
            "fever": polarity("fever"),
            "pmh_availability": {
                "labels": {
                    "documented": "PMH listed with conditions",
                    "not_available": "PMH unavailable / unknown / not obtained",
                    "none_significant": "no significant past medical history",
                }
            },
            "history_quality": {
                "labels": {
                    "usable": "enough history to triage",
                    "need_more": "must ask for missing history",
                }
            },
            "disposition": {
                "labels": {
                    "ed_now": "send to emergency department immediately",
                    "urgent_clinic": "same-day clinic or urgent care",
                    "routine": "non-urgent outpatient follow-up",
                    "self_care": "home care with return precautions",
                }
            },
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
        "min_confidence": 0.4,
        "focus": "affirmed chest pain, negated fever, documented PMH, ED disposition",
    },
    {
        "id": "medical_family_hx_only",
        "domain": "clinical_nlp",
        "title": "Family history only (not patient diagnosis)",
        "text": (
            "Family history is significant for colon cancer in father at age 55. "
            "Patient has no personal history of malignancy. Review of systems otherwise negative."
        ),
        "tasks": {
            "colon_cancer": {
                "labels": {
                    "patient_current": "patient has colon cancer now",
                    "patient_past": "patient had colon cancer before",
                    "family_only": "relative has colon cancer, not the patient",
                    "denied_patient": "patient denies personal malignancy history",
                    "not_mentioned": "colon cancer not discussed",
                }
            },
            "pmh_availability": {
                "labels": {
                    "documented": "PMH listed with conditions",
                    "not_available": "PMH unavailable / unknown / not obtained",
                    "none_significant": "no significant past medical history",
                    "family_only": "only family history is given",
                }
            },
            "next_action": {
                "labels": {
                    "extract_ok": "store as family history, not patient dx",
                    "ask_history": "still need patient PMH",
                    "escalate_llm": "hand off to LLM or clinician",
                }
            },
        },
        "min_confidence": 0.4,
        "focus": "family vs patient attribution",
    },
    {
        "id": "ecommerce",
        "domain": "ecommerce",
        "title": "Refund + shipping complaint",
        "text": "I was charged twice for order #4412 and the package still has not shipped. Refund the duplicate today.",
        "tasks": {
            "intent": ["refund_request", "order_status", "cancel_subscription", "return_item", "other"],
            "channel": ["billing", "shipping", "product_quality", "account"],
            "urgency": ["0", "1", "2", "3"],
            "handoff": {
                "labels": {
                    "yes": "needs a human agent now",
                    "no": "can be handled by automation",
                }
            },
        },
        "min_confidence": 0.55,
    },
    {
        "id": "fintech",
        "domain": "fintech",
        "title": "Card freeze + fraud concern",
        "text": "Someone tried to use my debit card in another city. Freeze the card and send a new one.",
        "tasks": {
            "intent": {
                "labels": {
                    "freeze_card": "block the payment card immediately",
                    "dispute_charge": "challenge a transaction",
                    "balance_inquiry": "ask about available funds",
                    "transfer": "move money between accounts",
                    "other": "something else",
                }
            },
            "risk": ["0", "1", "2", "3"],
            "products": {
                "labels": ["debit_card", "credit_card", "checking", "wire", "mobile_app"],
                "multi_label": True,
                "threshold": 0.4,
            },
        },
        "min_confidence": 0.55,
    },
    {
        "id": "edtech",
        "domain": "edtech",
        "title": "Gradebook issue before exam",
        "text": (
            "I submitted my calculus homework twice and the gradebook still shows zero. "
            "Midterm is tomorrow — can a human check this?"
        ),
        "tasks": {
            "intent": [
                "grade_issue",
                "content_question",
                "tech_support",
                "deadline_extension",
                "other",
            ],
            "subject": ["math", "science", "english", "history", "computer_science", "other"],
            "needs_human": ["yes", "no"],
            "urgency": ["0", "1", "2", "3"],
        },
        "min_confidence": 0.55,
    },
    {
        "id": "ops_urgency",
        "domain": "ops",
        "title": "Incident urgency",
        "text": "The deploy failed twice and customers are seeing 500s. Can someone look now?",
        "tasks": {
            "urgent": {
                "labels": {
                    "yes": "needs attention right now",
                    "no": "this can wait",
                }
            },
            "severity": ["0", "1", "2", "3"],
            "area": ["infrastructure", "application", "security", "data", "other"],
        },
        "min_confidence": 0.55,
    },
    {
        "id": "clinical_triage",
        "domain": "clinical",
        "title": "Chest-pain intake (schema demo)",
        "text": (
            "52M with sudden crushing chest pain radiating to left arm for 40 minutes, "
            "diaphoresis, history of hypertension. BP 168/102, HR 110. No fever."
        ),
        "tasks": {
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
        "min_confidence": 0.55,
        "warn": "Not a medical device.",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/onesystem", help="local checkpoint directory")
    parser.add_argument("--out", default="docs/data/recipes.json", help="write results JSON")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from onesystem.model import OneSystem

    model_path = Path(args.model)
    if not (model_path / "model.safetensors").is_file():
        print(
            f"ERROR: {model_path} has no model.safetensors.\n"
            "Offline use requires weights already on disk "
            "(copy from GitHub release v0.3.0 into models/onesystem/).",
            file=sys.stderr,
        )
        return 2

    print(f"Loading {model_path} …", flush=True)
    model = OneSystem.load(str(model_path), device=args.device)

    results = []
    for case in CASES:
        print(f"\n=== {case['id']}: {case['title']} ===", flush=True)
        print(f"TEXT: {case['text']}", flush=True)
        out = model.classify(case["text"], case["tasks"], min_confidence=case.get("min_confidence"))
        print(json.dumps(out, indent=2), flush=True)
        results.append(
            {
                "id": case["id"],
                "domain": case["domain"],
                "title": case["title"],
                "text": case["text"],
                "focus": case.get("focus"),
                "warn": case.get("warn"),
                "min_confidence": case.get("min_confidence"),
                "output": out,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(model_path),
        "name": model.name,
        "disclaimer": (
            "Clinical / medical schemas are illustrative only and are not a medical device. "
            "For production clinical NLP, fine-tune with domain_adapt on labelled notes."
        ),
        "cases": results,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
