#!/usr/bin/env python3
"""Build a healthcare text corpus for MedLM pretrain.

Sources (combined by default):
  1) Local clinic-style JSONL already in the repo
  2) Synthetic English bootstrap (small, disk-safe)
  3) Hugging Face ``peoplemaketh/humigy-tiny-data`` (clinical phrases / notes,
     CC-BY-4.0) — downloaded via huggingface_hub (avoids broken datasets cast)

Disk-conscious: humigy jsonl totals ~6 MB. Drop more licensed corpora into
data/medlm/raw/ to scale.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HUMIGY_REPO = "peoplemaketh/humigy-tiny-data"


BOOTSTRAP_TEMPLATES = [
    "Patient presents with {symptom}. Vital signs: BP {bp}, HR {hr}, Temp {temp} F. "
    "Denies {denied}. History of {hx}. Plan: {plan}.",
    "Chief complaint: {symptom}. Onset {onset}. Associated symptoms include {assoc}. "
    "No {denied}. Past medical history: {hx}. Assessment and plan: {plan}.",
    "US outpatient note. Reason for visit: {symptom}. Review of systems negative for {denied}. "
    "Medications reviewed. Follow-up: {plan}.",
    "Emergency triage summary. Primary symptom {symptom} for {onset}. "
    "Blood pressure {bp}, heart rate {hr}. History notable for {hx}. Next step: {plan}.",
    "Care coordination update. Patient reports {symptom}. "
    "Allergy list reviewed. Denied {denied}. Scheduled {plan}.",
    "Discharge instructions. Diagnosis-related counseling for {hx}. "
    "Return precautions if {symptom} worsens. Activity: rest. Follow up with primary care for {plan}.",
    "Medication reconciliation. Continue current therapy for {hx}. "
    "Monitor for {symptom}. Call clinic if {denied} develops unexpectedly.",
    "Telehealth visit. Patient describes {symptom} over {onset}. "
    "No chest pain or dyspnea unless noted. History: {hx}. Plan includes {plan}.",
]

SYMPTOMS = [
    "cough", "fatigue", "knee pain", "headache", "nausea", "back pain",
    "sore throat", "dizziness", "rash", "abdominal discomfort", "insomnia",
    "joint stiffness", "shortness of breath on exertion", "ear pain",
]
DENIED = [
    "chest pain", "fever", "vomiting", "syncope", "bloody stool",
    "vision loss", "severe dyspnea", "allergic reaction",
]
HX = [
    "hypertension", "type 2 diabetes", "asthma", "hyperlipidemia",
    "osteoarthritis", "hypothyroidism", "seasonal allergies", "GERD",
    "no significant past medical history",
]
PLAN = [
    "supportive care and hydration",
    "lab review at follow-up",
    "physical therapy referral",
    "medication refill after chart review",
    "return precautions and primary care follow-up in 1 week",
    "imaging if symptoms persist beyond 7 days",
    "nurse callback for education",
]
ONSET = ["1 day", "3 days", "1 week", "2 weeks", "several hours", "since yesterday"]
ASSOC = [
    "mild congestion", "low energy", "poor sleep", "activity limitation",
    "intermittent discomfort", "reduced appetite",
]


def _bp(rng: random.Random) -> str:
    return f"{rng.randint(105, 158)}/{rng.randint(68, 98)}"


def generate_bootstrap(n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        tmpl = rng.choice(BOOTSTRAP_TEMPLATES)
        text = tmpl.format(
            symptom=rng.choice(SYMPTOMS),
            denied=rng.choice(DENIED),
            hx=rng.choice(HX),
            plan=rng.choice(PLAN),
            onset=rng.choice(ONSET),
            assoc=rng.choice(ASSOC),
            bp=_bp(rng),
            hr=str(rng.randint(58, 110)),
            temp=f"{rng.uniform(97.6, 100.4):.1f}",
        )
        rows.append(f"Document {i+1}. {text}")
    return rows


def load_repo_medical_text(repo: Path) -> list[str]:
    texts: list[str] = []
    candidates = [
        repo / "data" / "raw" / "clinic_request.jsonl",
        repo / "data" / "raw" / "benefits_request.jsonl",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw = row.get("input") or ""
                raw = re.sub(r"\[channel\]\s*\w+", " ", raw)
                raw = re.sub(r"\[(message|utterance)\]\s*", " ", raw)
                raw = re.sub(r"\s+", " ", raw).strip()
                if len(raw) > 40:
                    texts.append(raw)
    return texts


def _format_humigy_row(row: dict) -> str:
    """Turn a humigy JSON object into a plain pretrain document."""
    text = (row.get("text") or "").strip()
    if not text:
        return ""
    parts = [text]
    clinic = row.get("clinic")
    style = row.get("style")
    kind = row.get("kind")
    meta = []
    if clinic:
        meta.append(f"clinic={clinic}")
    if style:
        meta.append(f"style={style}")
    if kind:
        meta.append(f"kind={kind}")
    if meta:
        parts.append("(" + ", ".join(meta) + ")")
    for key, label in (("dx", "dx"), ("labs", "labs"), ("med", "med"), ("must_not", "must_not")):
        vals = row.get(key) or []
        if isinstance(vals, list) and vals:
            parts.append(f"{label}: " + "; ".join(str(v) for v in vals if v))
        elif isinstance(vals, str) and vals.strip():
            parts.append(f"{label}: {vals.strip()}")
    return " | ".join(parts)


def download_humigy(cache_dir: Path) -> Path:
    """Snapshot jsonl files from HF (no ``datasets`` cast — hub only)."""
    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=HUMIGY_REPO,
        repo_type="dataset",
        local_dir=str(cache_dir),
        allow_patterns=["**/*.jsonl", "README.md"],
    )
    return Path(path)


def load_humigy_texts(cache_dir: Path, *, max_docs: int | None = None) -> list[str]:
    texts: list[str] = []
    for path in sorted(cache_dir.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                doc = _format_humigy_row(row)
                if len(doc) >= 8:
                    texts.append(doc)
                if max_docs is not None and len(texts) >= max_docs:
                    return texts
    return texts


def _texts_from_jsonl_dir(root: Path) -> list[str]:
    texts: list[str] = []
    if not root.is_dir():
        return texts
    for path in sorted(root.rglob("*.txt")):
        body = path.read_text(encoding="utf-8", errors="ignore").strip()
        if body:
            texts.extend(p.strip() for p in body.split("\n\n") if len(p.strip()) > 20)
    for path in sorted(root.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    for key in ("text", "input", "content", "document"):
                        if isinstance(row.get(key), str) and len(row[key].strip()) > 20:
                            texts.append(row[key].strip())
                            break
    return texts


def load_extra_raw(raw_dir: Path) -> list[str]:
    """Extra corpora: data/medlm/raw/sources/ and data/medlm/public/."""
    texts = _texts_from_jsonl_dir(raw_dir / "sources")
    public = ROOT / "data" / "medlm" / "public"
    texts.extend(_texts_from_jsonl_dir(public))
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/medlm/raw")
    parser.add_argument("--bootstrap", type=int, default=4000, help="synthetic docs (0 to disable)")
    parser.add_argument("--humigy", action=argparse.BooleanOptionalAction, default=True,
                        help="download + include peoplemaketh/humigy-tiny-data")
    parser.add_argument("--humigy-dir", default="data/medlm/humigy-tiny-data")
    parser.add_argument("--humigy-max", type=int, default=None, help="cap humigy docs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    local = load_repo_medical_text(ROOT)
    boot = generate_bootstrap(args.bootstrap, seed=args.seed) if args.bootstrap > 0 else []
    humigy: list[str] = []
    if args.humigy:
        print(f"Downloading {HUMIGY_REPO} …", flush=True)
        cache = download_humigy(Path(args.humigy_dir))
        humigy = load_humigy_texts(cache, max_docs=args.humigy_max)
        print(f"humigy docs: {len(humigy)}", flush=True)

    extra = load_extra_raw(out)
    all_texts = local + boot + humigy + extra
    # de-dupe exact strings
    all_texts = list(dict.fromkeys(all_texts))
    rng = random.Random(args.seed)
    rng.shuffle(all_texts)

    n = len(all_texts)
    if n < 50:
        raise SystemExit("corpus too small — enable --humigy or raise --bootstrap")
    n_val = max(200, n // 20)
    val, train = all_texts[:n_val], all_texts[n_val:]

    train_path = out / "train.txt"
    val_path = out / "val.txt"
    train_path.write_text("\n\n".join(train) + "\n", encoding="utf-8")
    val_path.write_text("\n\n".join(val) + "\n", encoding="utf-8")

    meta = {
        "train_docs": len(train),
        "val_docs": len(val),
        "local_docs": len(local),
        "bootstrap_docs": len(boot),
        "humigy_docs": len(humigy),
        "extra_docs": len(extra),
        "humigy_repo": HUMIGY_REPO if args.humigy else None,
        "train_chars": train_path.stat().st_size,
        "val_chars": val_path.stat().st_size,
        "disclaimer": (
            "Includes synthetic bootstrap and CC-BY-4.0 humigy-tiny-data "
            "(peoplemaketh). Research pretraining only. Not a medical device."
        ),
    }
    (out / "corpus_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {train_path} and {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
