# OneSystem

OneSystem is a local specialist model for operational decisions. You pass a document and the labels that are legal for that decision. It returns the chosen label and a confidence score in one forward pass. It does not generate text.

The saved checkpoint is a LoRA adapter named **OneSystem**. It starts from the public [`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide) weights and is trained on the public [`fastino/fast-decisions`](https://huggingface.co/datasets/fastino/fast-decisions) development split. Twenty percent of each domain is held out of training. That holdout is not Fastino's unpublished test set, so a score in this repository is not the published fast-decisions benchmark.

## Install

Python 3.10 or newer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[local,dev]"
```

The `local` extra installs PyTorch, Transformers, and GLiNER2. The first load downloads the base checkpoint (about 1.9 GB) from Hugging Face.

## Decide something

```python
from onesystem.model import OneSystem

model = OneSystem.load()
print(model.classify(
    "Hi, we were billed twice for March. Please refund the duplicate today.",
    {"intent": ["refund_request", "order_status", "cancel_subscription", "other"]},
))
```

From the command line:

```bash
onesystem "Stop the bot and get me a person." --task handoff --labels yes,no
```

`OneSystem.load()` uses `models/onesystem` when that adapter is present, and otherwise the base Decide checkpoint.

## Train

Training reads the public development files, writes `data/train.jsonl` and `data/eval.jsonl`, and saves the adapter to `models/onesystem`.

```bash
python -m onesystem.train
python -m onesystem.evaluate
```

The default run is one epoch of LoRA (rank 8) on the encoder and classifier, sequence length 384, batch size 1. On this project's development machine that run uses the Apple GPU. The upstream trainer itself only selects CUDA or CPU; `onesystem.train` moves the model to MPS when CUDA is absent.

## Evaluation

`python -m onesystem.evaluate` scores exact label match on the local holdout and writes `models/onesystem/eval.json`. Report that file with the holdout note. Do not present it as Fastino's 300-example-per-domain test score.

The saved OneSystem adapter, one epoch on the Apple GPU, scores **0.679 exact match** on that holdout: 394 of 580 decision heads across 340 rows. Single-label heads such as handoff and sentiment are stronger. Multi-label heads remain the weak spot, because a head counts as correct only when the full label set matches.

## Repository layout

| Path | Purpose |
|---|---|
| `onesystem/model.py` | Load the adapter and classify |
| `onesystem/data.py` | Download, validate, and split the public development data |
| `onesystem/train.py` | LoRA training and the `models/onesystem` save |
| `onesystem/evaluate.py` | Holdout exact-match score |
| `tests/` | Unit tests that do not need a GPU |
| `models/onesystem` | Saved OneSystem adapter, written by training |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests are welcome. Useful contributions include a new labelled domain, a training fix, or an evaluation that stays on untouched data.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The base weights and the public development data remain under their own Apache 2.0 releases from Fastino. OneSystem does not redistribute those base weights; the adapter stores only the LoRA update.

## Limitations

- The public development split is 1,700 rows across 17 domains. That is enough for this adapter and too small to reproduce the Decide checkpoint.
- Choice quality drops when a label set is large or the labels are easy to confuse. Prefer a short candidate list, and include an `other` label when the set can miss the real case.
- Confidence is the model's score for the selected label. Calibrate it on your own data before you branch production traffic on it.
- English is the training language of the base checkpoint used here.
