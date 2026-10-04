# Public benchmarks

Scripts and JSON results for **BANKING77**, **CLINC150**, **HWU64**, and **GoEmotions**.

## Run

```bash
# data downloads under data/benchmarks/ (gitignored)
python benchmarks/run_banking77.py --epochs 4 --batch-size 8
python benchmarks/run_clinc150.py --epochs 3 --batch-size 8
python benchmarks/run_hwu64.py --epochs 4 --batch-size 8
python benchmarks/run_goemotions.py --epochs 2 --batch-size 8
```

Protocol:

- Every example sees the **full** label set for that dataset.
- Label wording = publisher category strings (frozen).
- Calibration / abstain threshold from train/val only — **official test untouched**.
- Two tracks: current OneSystem release checkpoint, and a domain-adapted fine-tune.

## Your own domain

```bash
python -m onesystem.domain_adapt \
  --train data/my_domain/train.jsonl \
  --output models/onesystem-mydomain \
  --name OneSystem-MyDomain \
  --epochs 4
```

See the [cookbook § Domain-specific model](https://kiranbeethoju.github.io/onesystem/#domain-adapt).

Published page: https://kiranbeethoju.github.io/onesystem/benchmarks.html  
Raw JSON mirrored in `docs/data/` for GitHub Pages.
