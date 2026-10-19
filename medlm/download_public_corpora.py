#!/usr/bin/env python3
"""Download **public** medical text corpora into data/medlm/public/.

Credentialed PhysioNet / n2c2 / MIMIC* sources are listed but NOT downloaded
(they need your PhysioNet login + DUA). Run those yourself after approval.

Usage:
  python -m medlm.download_public_corpora
  python -m medlm.download_public_corpora --only pubmedqa,medmcqa,humigy
  python -m medlm.download_public_corpora --list

Colab tip: run this on the Colab runtime (more disk) before encode/train.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "data" / "medlm" / "public"

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

CREDENTIALED_SKIP = [
    ("MIMIC-III", "https://physionet.org/content/mimiciii/", "PhysioNet credentialed"),
    ("MIMIC-IV", "https://physionet.org/content/mimiciv/", "PhysioNet credentialed"),
    ("MIMIC-IV-Note", "https://physionet.org/content/mimic-iv-note/", "PhysioNet credentialed — best clinical notes"),
    ("MIMIC-IV-ED", "https://physionet.org/content/mimic-iv-ed/", "PhysioNet credentialed"),
    ("eICU", "https://physionet.org/content/eicu-crd/", "PhysioNet credentialed"),
    ("HiRID", "https://physionet.org/content/hirid/", "PhysioNet credentialed"),
    ("SICdb", "https://physionet.org/content/sicdb/", "PhysioNet credentialed"),
    ("PIC", "https://physionet.org/content/picdb/", "PhysioNet credentialed"),
    ("MedNLI", "https://physionet.org/content/mednli/", "PhysioNet credentialed"),
    ("MIMIC-CXR", "https://physionet.org/content/mimic-cxr/", "PhysioNet credentialed"),
    ("n2c2 / i2b2", "https://n2c2.dbmi.hms.harvard.edu/data-sets", "Per-dataset DUA; do not redistribute"),
    ("AmsterdamUMCdb", "https://amsterdammedicaldatascience.nl/", "Application required"),
    ("UMLS / SNOMED", "https://www.nlm.nih.gov/research/umls/", "Licence required"),
]

PUBLIC_LOADERS: Dict[str, str] = {}  # filled by decorators


def register(name: str):
    def deco(fn: Callable):
        PUBLIC_LOADERS[name] = fn
        return fn

    return deco


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            text = (row.get("text") or "").strip()
            if len(text) < 20:
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _http_get(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    print(f"  GET {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "onesystem-medlm"})
    with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    return dest


# ----- loaders -------------------------------------------------------------


@register("humigy")
def load_humigy(out: Path) -> dict:
    from medlm.prepare_corpus import HUMIGY_REPO, download_humigy, load_humigy_texts

    cache = download_humigy(ROOT / "data" / "medlm" / "humigy-tiny-data")
    texts = load_humigy_texts(cache)
    n = _write_jsonl(out / "humigy.jsonl", ({"text": t, "source": "humigy-tiny-data"} for t in texts))
    return {"docs": n, "repo": HUMIGY_REPO}


@register("pubmedqa")
def load_pubmedqa(out: Path) -> dict:
    """Expert-labelled PubMedQA only (skip pqa_artificial)."""
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows = []
    for row in ds:
        q = row.get("question") or ""
        ctx = row.get("context") or {}
        contexts = ctx.get("contexts") if isinstance(ctx, dict) else None
        ctx_text = " ".join(contexts) if isinstance(contexts, list) else ""
        ans = row.get("long_answer") or row.get("final_decision") or ""
        text = f"Question: {q}\nContext: {ctx_text}\nAnswer: {ans}".strip()
        rows.append({"text": text, "source": "pubmedqa_pqa_labeled"})
    n = _write_jsonl(out / "pubmedqa_labeled.jsonl", rows)
    return {"docs": n, "note": "pqa_labeled only — artificial split excluded"}


@register("medmcqa")
def load_medmcqa(out: Path) -> dict:
    from datasets import load_dataset

    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    rows = []
    for row in ds:
        q = row.get("question") or ""
        opts = [row.get(f"opa"), row.get("opb"), row.get("opc"), row.get("opd")]
        opts = [o for o in opts if o]
        exp = row.get("exp") or ""
        text = f"Medical exam question: {q}\nOptions: " + " | ".join(opts)
        if exp:
            text += f"\nExplanation: {exp}"
        rows.append({"text": text, "source": "medmcqa"})
    n = _write_jsonl(out / "medmcqa.jsonl", rows)
    return {"docs": n}


@register("medqa")
def load_medqa(out: Path) -> dict:
    """MedQA USMLE-style via Hugging Face mirror when available."""
    from datasets import load_dataset

    tried = []
    for name, config in (
        ("GBaker/MedQA-USMLE-4-options", None),
        ("openlifescienceai/med_qa", None),
    ):
        try:
            kwargs = {"path": name, "split": "train"}
            if config:
                ds = load_dataset(name, config, split="train")
            else:
                ds = load_dataset(name, split="train")
            rows = []
            for row in ds:
                q = row.get("question") or row.get("sent1") or ""
                opts = row.get("options") or row.get("ending") or {}
                if isinstance(opts, dict):
                    opt_txt = " | ".join(f"{k}: {v}" for k, v in opts.items())
                elif isinstance(opts, list):
                    opt_txt = " | ".join(str(x) for x in opts)
                else:
                    opt_txt = str(opts)
                ans = row.get("answer") or row.get("answer_idx") or ""
                text = f"MedQA: {q}\nOptions: {opt_txt}\nAnswer: {ans}"
                rows.append({"text": text, "source": name})
            n = _write_jsonl(out / "medqa.jsonl", rows)
            return {"docs": n, "hf": name}
        except Exception as exc:  # noqa: BLE001
            tried.append(f"{name}: {exc}")
    raise RuntimeError("MedQA download failed: " + " ; ".join(tried))


@register("medicationqa")
def load_medicationqa(out: Path) -> dict:
    url = "https://raw.githubusercontent.com/abachaa/Medication_QA_MedInfo2019/master/MedInfo2019-QA-Medications.csv"
    # fallback raw paths vary; try HF if present
    try:
        from datasets import load_dataset

        ds = load_dataset("bigbio/medication_qa", "medication_qa_source", split="train")
        rows = []
        for row in ds:
            q = row.get("question") or ""
            a = row.get("answer") or row.get("response") or ""
            rows.append({"text": f"Medication question: {q}\nAnswer: {a}", "source": "medicationqa"})
        n = _write_jsonl(out / "medicationqa.jsonl", rows)
        if n:
            return {"docs": n, "via": "bigbio"}
    except Exception:
        pass
    # GitHub raw JSON/CSV mirrors
    for raw in (
        "https://raw.githubusercontent.com/abachaa/Medication_QA_MedInfo2019/master/MedInfo2019-QA-Medications.json",
        "https://huggingface.co/datasets/bigbio/medication_qa/resolve/main/medication_qa.json",
    ):
        try:
            dest = out / "_tmp_medicationqa"
            _http_get(raw, dest)
            text = dest.read_text(encoding="utf-8", errors="ignore")
            rows = []
            if text.lstrip().startswith("["):
                data = json.loads(text)
                for row in data:
                    q = row.get("Question") or row.get("question") or ""
                    a = row.get("Answer") or row.get("answer") or ""
                    rows.append({"text": f"Medication question: {q}\nAnswer: {a}", "source": "medicationqa"})
            n = _write_jsonl(out / "medicationqa.jsonl", rows)
            dest.unlink(missing_ok=True)
            if n:
                return {"docs": n, "via": raw}
        except Exception:
            continue
    return {"docs": 0, "error": "medicationqa unavailable"}


@register("liveqa")
def load_liveqa(out: Path) -> dict:
    try:
        from datasets import load_dataset

        ds = load_dataset("bigbio/liveqa", "liveqa_source", split="test")
        rows = []
        for row in ds:
            q = row.get("question") or row.get("message") or ""
            a = row.get("answer") or row.get("summary") or ""
            if not q and isinstance(row.get("nid"), str):
                continue
            text = f"Consumer health question: {q}\nAnswer: {a}".strip()
            rows.append({"text": text, "source": "liveqa"})
        n = _write_jsonl(out / "liveqa.jsonl", rows)
        return {"docs": n}
    except Exception as exc:  # noqa: BLE001
        return {"docs": 0, "error": str(exc)}


@register("pmc_patients")
def load_pmc_patients(out: Path) -> dict:
    from datasets import load_dataset

    # prefer HF mirrors
    for name in ("zhengyun21/PMC-Patients", "pmc/patients"):
        try:
            ds = load_dataset(name, split="train")
            rows = []
            for row in ds:
                t = row.get("patient") or row.get("text") or row.get("context") or ""
                title = row.get("title") or ""
                text = f"{title}\n{t}".strip() if title else t
                rows.append({"text": text, "source": "pmc_patients"})
            n = _write_jsonl(out / "pmc_patients.jsonl", rows)
            return {"docs": n, "hf": name}
        except Exception:
            continue
    return {"docs": 0, "error": "pmc_patients not found on HF"}


@register("bc5cdr")
def load_bc5cdr(out: Path) -> dict:
    from datasets import load_dataset

    ds = load_dataset("bigbio/bc5cdr", "bc5cdr_source", split="train", trust_remote_code=True)
    rows = []
    for row in ds:
        passages = row.get("passages") or row.get("text") or ""
        if isinstance(passages, list):
            text = "\n".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in passages
            )
        else:
            text = str(passages)
        rows.append({"text": text, "source": "bc5cdr"})
    n = _write_jsonl(out / "bc5cdr.jsonl", rows)
    return {"docs": n}


@register("ncbi_disease")
def load_ncbi_disease(out: Path) -> dict:
    from datasets import load_dataset

    for name in ("bigbio/ncbi_disease", "ncbi/ncbi_disease"):
        try:
            ds = load_dataset(name, "ncbi_disease_source", split="train", trust_remote_code=True)
            rows = []
            for row in ds:
                passages = row.get("passages") or []
                if passages:
                    text = "\n".join(p.get("text", "") for p in passages if isinstance(p, dict))
                else:
                    text = str(row.get("text") or "")
                rows.append({"text": text, "source": "ncbi_disease"})
            n = _write_jsonl(out / "ncbi_disease.jsonl", rows)
            return {"docs": n, "hf": name}
        except Exception:
            continue
    return {"docs": 0, "error": "ncbi_disease failed"}


@register("biosses")
def load_biosses(out: Path) -> dict:
    from datasets import load_dataset

    ds = load_dataset("bigbio/biosses", "biosses_source", split="train", trust_remote_code=True)
    rows = []
    for row in ds:
        a = row.get("text_1") or row.get("sentence1") or ""
        b = row.get("text_2") or row.get("sentence2") or ""
        rows.append({"text": f"{a}\n{b}".strip(), "source": "biosses"})
    n = _write_jsonl(out / "biosses.jsonl", rows)
    return {"docs": n}


@register("headqa")
def load_headqa(out: Path) -> dict:
    from datasets import load_dataset

    try:
        ds = load_dataset("head_qa", "en", split="train")
    except Exception:
        ds = load_dataset("head_qa", split="train")
    rows = []
    for row in ds:
        q = row.get("qtext") or row.get("question") or ""
        answers = row.get("answers") or []
        if isinstance(answers, list):
            opt = " | ".join(
                (a.get("atext") if isinstance(a, dict) else str(a)) for a in answers
            )
        else:
            opt = str(answers)
        rows.append({"text": f"HEAD-QA: {q}\nOptions: {opt}", "source": "headqa"})
    n = _write_jsonl(out / "headqa.jsonl", rows)
    return {"docs": n}


@register("cord19_meta")
def load_cord19_meta(out: Path) -> dict:
    """CORD-19 metadata titles+abstracts (compact; not full PDFs)."""
    from datasets import load_dataset

    try:
        ds = load_dataset("allenai/cord19", "metadata", split="train", trust_remote_code=True)
    except Exception:
        ds = load_dataset("cord19", "metadata", split="train", trust_remote_code=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= 50000:  # cap for disk
            break
        title = row.get("title") or ""
        abstract = row.get("abstract") or ""
        text = f"{title}\n{abstract}".strip()
        rows.append({"text": text, "source": "cord19_metadata"})
    n = _write_jsonl(out / "cord19_meta.jsonl", rows)
    return {"docs": n, "capped": 50000}


@register("pubmed_abstracts_sample")
def load_pubmed_sample(out: Path) -> dict:
    """Small public PubMed abstract sample from HF (not full MEDLINE dump)."""
    from datasets import load_dataset

    for name, split in (
        ("pubmed_qa", "pqa_unlabeled"),  # uses real abstracts — not artificial QA
        ("ccdv/pubmed-summarization", "train"),
    ):
        try:
            if name == "pubmed_qa":
                ds = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split="train")
                rows = []
                for i, row in enumerate(ds):
                    if i >= 30000:
                        break
                    q = row.get("question") or ""
                    ctx = row.get("context") or {}
                    contexts = ctx.get("contexts") if isinstance(ctx, dict) else []
                    ctx_text = " ".join(contexts) if contexts else ""
                    # unlabeled: keep literature text, drop fake yes/no labels
                    text = f"{q}\n{ctx_text}".strip()
                    rows.append({"text": text, "source": "pubmedqa_unlabeled_context"})
                n = _write_jsonl(out / "pubmed_contexts.jsonl", rows)
                return {"docs": n, "note": "contexts from pqa_unlabeled (not artificial answers)"}
            ds = load_dataset(name, split=split)
            rows = []
            for i, row in enumerate(ds):
                if i >= 20000:
                    break
                art = row.get("article") or row.get("text") or ""
                rows.append({"text": art, "source": name})
            n = _write_jsonl(out / "pubmed_sum.jsonl", rows)
            return {"docs": n, "hf": name}
        except Exception:
            continue
    return {"docs": 0, "error": "pubmed sample failed"}


DEFAULT_SET = [
    "humigy",
    "pubmedqa",
    "medmcqa",
    "medqa",
    "medicationqa",
    "liveqa",
    "pmc_patients",
    "bc5cdr",
    "ncbi_disease",
    "biosses",
    "headqa",
    "pubmed_abstracts_sample",
    # cord19_meta is larger — include unless --fast
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_ROOT))
    parser.add_argument("--only", default=None, help="comma-separated loader names")
    parser.add_argument("--fast", action="store_true", help="skip larger optional sets")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--skip-credentialed-report", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Public loaders:")
        for name in sorted(PUBLIC_LOADERS):
            print(" ", name)
        print("\nCredentialed (NOT auto-downloaded):")
        for name, url, note in CREDENTIALED_SKIP:
            print(f"  - {name}: {note}\n    {url}")
        return 0

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("Installing datasets …", flush=True)
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = (
        [x.strip() for x in args.only.split(",") if x.strip()]
        if args.only
        else list(DEFAULT_SET)
    )
    if args.fast:
        names = [n for n in names if n not in {"cord19_meta", "pmc_patients", "pubmed_abstracts_sample"}]

    summary = {"public": {}, "credentialed_skipped": [
        {"name": n, "url": u, "note": note} for n, u, note in CREDENTIALED_SKIP
    ]}
    for name in names:
        fn = PUBLIC_LOADERS.get(name)
        if not fn:
            print(f"UNKNOWN {name}", flush=True)
            continue
        print(f"==> {name}", flush=True)
        try:
            info = fn(out / name)
        except Exception as exc:  # noqa: BLE001
            info = {"docs": 0, "error": str(exc)}
            print(f"  ERROR: {exc}", flush=True)
        summary["public"][name] = info
        print(f"  {info}", flush=True)

    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(summary, indent=2) + "\n")
    total = sum(int(v.get("docs") or 0) for v in summary["public"].values())
    print(f"\nTotal public docs written: {total}")
    print(f"Manifest: {manifest}")
    if not args.skip_credentialed_report:
        print("\nCredentialed sources were NOT downloaded. Approve on PhysioNet/n2c2, then place files under data/medlm/raw/sources/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
