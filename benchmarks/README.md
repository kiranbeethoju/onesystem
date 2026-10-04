# Public benchmarks

Scripts and JSON results for **BANKING77** and **CLINC150**.

## Run

```bash
# data already under data/benchmarks/ (gitignored) — re-download if needed
python benchmarks/run_banking77.py --epochs 4 --batch-size 8
python benchmarks/run_clinc150.py --epochs 3 --batch-size 8
```

Protocol:

- Every example sees the **full** label set (77 / 150).
- Label wording = publisher category strings (frozen).
- Calibration / abstain threshold from train/val only — **official test untouched**.
- Two tracks: current OneSystem release checkpoint, and a domain-adapted fine-tune.

Published page: https://kiranbeethoju.github.io/onesystem/benchmarks.html  
Raw JSON mirrored in `docs/data/` for GitHub Pages.
