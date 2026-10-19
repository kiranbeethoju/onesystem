# MedLM — ~120M from-scratch healthcare LM (Mac)

Pretrain a **decoder-only Transformer (~100–130M params)** from **random initialization** on compact US-healthcare-style English.

**Not a medical device.** Synthetic bootstrap + local clinic-routing text only unless you add licensed corpora.

## Hardware note (this machine class)

| Constraint | Guidance |
|---|---|
| Apple Silicon, **18 GB** unified | Use defaults: `batch_size=1–2`, `grad_accum=8–16`, `block_size=512` |
| Tight disk | Keep one checkpoint; expand corpus later under `data/medlm/raw/` |
| MPS | Default device when available |

A *serious* medical LM needs **many GB of licensed text** and longer training than a laptop smoke run.

## Smoke run (M3 Pro 18 GB, MPS) — done

| Metric | Value |
|---|---|
| Params | **124.24M** (16 layer × 768, vocab 13742) |
| Steps | 300 |
| Train loss | 9.71 → ~0.96 |
| Val loss | **1.30** |
| Checkpoint | `models/medlm-120m/` (~475 MB `model.pt`) |

Corpus was compact (~1.6 MB text: local clinic-style rows + synthetic bootstrap). Loss drops fast because the bootstrap is repetitive — **add real licensed corpora before claiming a domain LM**.

## Pipeline

```bash
# from repo root, with torch + tokenizers available (transformers stack)
export PYTHONPATH=$PWD

# Default corpus = local clinic JSONL + bootstrap + Hugging Face humigy-tiny-data
# https://huggingface.co/datasets/peoplemaketh/humigy-tiny-data  (CC-BY-4.0)
python -m medlm.prepare_corpus --bootstrap 4000 --humigy
# python -m medlm.prepare_corpus --no-humigy   # offline / no Hub

python -m medlm.build_tokenizer --vocab-size 16000
python -m medlm.encode_data
python -m medlm.train --max-steps 500 --batch-size 1 --grad-accum 8 --save-every 250 --eval-every 100

# continue toward a fuller pretrain (after adding data, re-run encode_data)
python -m medlm.train --max-steps 5000 --batch-size 1 --grad-accum 16 --resume models/medlm-120m

# sample
python -m medlm.sample --model models/medlm-120m --prompt "Patient presents with cough."
```

### External datasets

| Dataset | Role |
|---|---|
| [`peoplemaketh/humigy-tiny-data`](https://huggingface.co/datasets/peoplemaketh/humigy-tiny-data) | Clinical phrases / handwritten-style notes (EN + Hinglish) |
| `python -m medlm.download_public_corpora` | PubMedQA, MedMCQA, MedQA, PMC-Patients, BigBIO sets, … → `data/medlm/public/` |
| `data/medlm/raw/sources/` | Credentialed MIMIC / n2c2 uploads (not auto-downloaded) |

**Colab GPU (T4/L4/A100):** see [`medlm/COLAB.md`](COLAB.md)

```bash
python -m medlm.download_public_corpora --list          # what is public vs credentialed
python -m medlm.download_public_corpora                 # fetch public → data/medlm/public/
python -m medlm.prepare_corpus --bootstrap 2000 --humigy
```

## Layout

| Path | Role |
|---|---|
| `medlm/model.py` | GPT-style decoder, random init |
| `medlm/train.py` | AdamW pretrain loop (MPS/CPU) |
| `medlm/prepare_corpus.py` | local + synthetic corpus |
| `medlm/build_tokenizer.py` | BPE tokenizer |
| `medlm/encode_data.py` | pack tokens to `.bin` |
| `models/medlm-120m/` | checkpoints (gitignored) |

## Scale up

1. Drop more `*.txt` / `*.jsonl` into `data/medlm/raw/` (licensed PubMed, guidelines, de-identified notes if allowed).  
2. Retokenize + re-encode.  
3. Raise `--max-steps` (tens of thousands+).  
4. Prefer **64 GB** unified memory for comfortable 200M runs.
