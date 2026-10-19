# Train MedLM-120M on Google Colab (T4 / L4 / A100)

## 1) Runtime

**Runtime → Change runtime type**

| Field | Recommended |
|---|---|
| Runtime type | Python 3 |
| Hardware accelerator | **L4** or **A100** (T4 also works) |
| High-RAM | On if available |
| Runtime version | Latest |

Click **Save**, then wait for the runtime to connect.

## 2) One-shot setup + download + train

Paste into a Colab cell:

```python
# clone
!git clone --depth 1 -b onesystem https://github.com/kiranbeethoju/onesystem.git
%cd /content/onesystem
!pip -q install -U "torch" "datasets" "tokenizers" "huggingface_hub" "safetensors" "tqdm"

import os, torch
os.environ["PYTHONPATH"] = "/content/onesystem"
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
```

```python
# public corpora → data/medlm/public/
# (skips MIMIC / n2c2 / PhysioNet — those need YOUR credentialed download)
!python -m medlm.download_public_corpora --out data/medlm/public
```

```python
# pack corpus
!python -m medlm.prepare_corpus --bootstrap 2000 --humigy
!python -m medlm.build_tokenizer --vocab-size 16000
!python -m medlm.encode_data
```

### GPU train presets

**T4 (16 GB)**

```python
!python -m medlm.train --device cuda --max-steps 10000 \
  --batch-size 4 --grad-accum 8 --block-size 512 \
  --warmup-steps 200 --eval-every 200 --save-every 500 \
  --out models/medlm-120m
```

**L4 (24 GB)**

```python
!python -m medlm.train --device cuda --max-steps 10000 \
  --batch-size 8 --grad-accum 4 --block-size 512 \
  --warmup-steps 200 --eval-every 200 --save-every 500 \
  --out models/medlm-120m
```

**A100 (40 GB)**

```python
!python -m medlm.train --device cuda --max-steps 20000 \
  --batch-size 16 --grad-accum 4 --block-size 512 \
  --warmup-steps 200 --eval-every 200 --save-every 500 \
  --out models/medlm-120m
```

## 3) Sample + download weights

```python
!python -m medlm.sample --model models/medlm-120m --device cuda \
  --prompt "Patient presents with fever and cough." --tokens 80
!zip -r /content/medlm-120m.zip models/medlm-120m
```

Then download `/content/medlm-120m.zip` from the Colab file browser.

## 4) Credentialed data (optional, after PhysioNet approval)

```text
Upload MIMIC-IV-Note / n2c2 extracts → /content/onesystem/data/medlm/raw/sources/
Then re-run prepare_corpus + encode_data + train --resume models/medlm-120m
```

## 5) What auto-download includes vs skips

| Auto (public) | Skipped (you must approve) |
|---|---|
| humigy-tiny-data, PubMedQA labeled, MedMCQA, MedQA, MedicationQA, LiveQA, PMC-Patients, BC5CDR, NCBI Disease, BIOSSES, HEAD-QA, PubMed context sample | MIMIC-III/IV/Note/ED/CXR, eICU, n2c2, MedNLI, AmsterdamUMCdb, UMLS/SNOMED |

See `python -m medlm.download_public_corpora --list`
