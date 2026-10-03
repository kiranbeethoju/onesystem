# OneSystem

OneSystem is a standalone System 1 decision model. You pass a document and the labels that are legal for a decision; it returns the chosen label, a calibrated confidence, and the full probability table in one forward pass. It does not generate text, and it does not load any third-party decision model at runtime: the weights you download are OneSystem's own.

The model is trained in this repository on the public [`fastino/fast-decisions`](https://huggingface.co/datasets/fastino/fast-decisions) development split. Twenty percent of each domain is held out for evaluation and another ten percent for confidence calibration. That holdout is not Fastino's unpublished test set, so a score in this repository is not the published fast-decisions benchmark.

## Install

Python 3.10 or newer. Works on CPU, CUDA, and Apple MPS.

```bash
git clone https://github.com/kiranbeethoju/onesystem.git
cd onesystem
pip install -e .
```

In Google Colab:

```python
!git clone https://github.com/kiranbeethoju/onesystem.git
%cd onesystem
!pip install -q -e .

from onesystem.model import OneSystem
model = OneSystem.load(device="cuda")
model.classify(
    "Hi, we were billed twice for March. Please refund the duplicate today.",
    {"intent": ["refund_request", "order_status", "cancel_subscription", "other"]},
)
```

`OneSystem.load()` uses `models/onesystem` when the weights are present in the checkout, and otherwise downloads the OneSystem release assets (about 420 MB) from this repository's GitHub release into `~/.cache/onesystem`. Set `ONESYSTEM_HOME` to move the cache. There is no Hugging Face token, no `gliner2`, and no upstream checkpoint involved.

## Decide something

```python
from onesystem.model import OneSystem

model = OneSystem.load()

model.classify(
    "Stop the bot and get me a person.",
    {
        "handoff": ["yes", "no"],
        "sentiment": ["negative", "neutral", "positive"],
        "urgency": ["0", "1", "2", "3"],                      # ordinal: also returns "score"
        "areas": {"labels": ["billing", "mobile_app", "login"], "multi_label": True, "threshold": 0.5},
        "doc_type": {"labels": {"invoice": "a bill for goods", "contract": "a signed agreement"}},
    },
    min_confidence=0.6,                                       # abstain below this
)
```

A single-label head answers `{"label", "confidence", "probabilities"}`; ordinal heads add `"score"` (the expected level); a multi-label head answers `{"labels", "probabilities", "threshold"}`. With `min_confidence`, a head whose best probability is below the floor returns `"label": None, "abstain": True` so your code can route it to a person.

From the command line:

```bash
onesystem "Stop the bot and get me a person." --task handoff --labels yes,no
onesystem "Checkout crashed and I never got a receipt." --task areas --labels checkout,notifications,login --multi-label
```

## What OneSystem brings

| | OneSystem |
|---|---|
| Own weights | The published `model.safetensors` is the full OneSystem network. Loading never touches another vendor's decision checkpoint. |
| Task-conditioned bi-encoder | Text and each candidate label are encoded with the task name as a prefix, passed through one shared projection, and scored by a learned-scale cosine. The same label set means the same label vectors, so a fixed schema costs one text pass per request. |
| Typed outputs | Single-label softmax, multi-label sigmoid with a per-task threshold, and an ordinal expected level for numeric scales. |
| Calibrated confidence | One temperature is fit on a calibration slice that is disjoint from both training and evaluation, and stored in the model config. The eval report includes expected calibration error, not just accuracy. |
| Abstention | `min_confidence` turns a weak decision into an explicit `abstain` instead of a guess. |
| Label descriptions | Pass `{label: description}` and the description is encoded with the label, which helps when label names are terse. |
| Zero-shot baseline in the report | Training logs the untrained network's score on the same holdout, so the gain from training is visible instead of implied. |
| Pure PyTorch and Transformers | No private training library. `python -m onesystem.train` is about three hundred lines you can read and change. |

### Architecture

```
text  --"{task}: {text}"-->  encoder --> masked mean --> proj --> normalize --\
                                                                               cosine * exp(log_scale) / temperature --> softmax | sigmoid
label --"{task}: {label}"--> encoder --> masked mean --> proj --> normalize --/
```

The encoder is a 110M-parameter BERT-style network. Its parameters were initialised from `BAAI/bge-base-en-v1.5` (MIT) before OneSystem training and are saved inside OneSystem; the published `config.json` embeds the encoder architecture so a load rebuilds the network from the OneSystem files alone. The shared projection starts as the identity, so the untrained network already decides by the encoder's own similarity (the zero-shot row in the manifest) and training sharpens that geometry instead of relearning it from two random spaces. Training uses eager attention because fused attention has no dropout kernel on Apple MPS.

## Train

Training downloads the public development files, writes `data/train.jsonl`, `data/calib.jsonl`, and `data/eval.jsonl`, and saves the full model, tokenizer, manifest, and evaluation to `models/onesystem`.

```bash
pip install -e ".[dev]"
python -m onesystem.train            # 6 epochs, batch 16; ~17 min on an M3 Pro, a few minutes on a T4
python -m onesystem.evaluate         # rescore models/onesystem on data/eval.jsonl
python -m onesystem.evaluate --fit-temperature   # refit calibration and save it
```

Options: `--epochs`, `--batch-size`, `--max-text-len`, `--encoder` (a different Hugging Face encoder to initialise from), `--output-dir`, `--skip-zero-shot`.

## Evaluation

`models/onesystem/eval.json` holds exact match and expected calibration error on the local holdout, per task. `models/onesystem/onesystem.json` records the data split, the zero-shot score before training, the training loss per epoch, the fitted temperature, and the final score. Report those files with the holdout note. Do not present them as Fastino's 300-example-per-domain test score.

Published release `v0.2.0`: zero-shot 0.453 exact match before training, **0.552 exact match (320 of 580 heads)** after training, expected calibration error 0.074 on the local holdout. Per-task numbers and the weak spots are in `models/onesystem/README.md`.

## Repository layout

| Path | Purpose |
|---|---|
| `onesystem/modeling.py` | `OneSystemConfig` and `OneSystemModel` (the network, save and load) |
| `onesystem/model.py` | `OneSystem`: task schema handling, label cache, `classify`, abstention |
| `onesystem/hub.py` | Resolve `models/onesystem` or download the GitHub release assets |
| `onesystem/data.py` | Download, validate, and split the public development data (train / calib / eval) |
| `onesystem/train.py` | Training loop, calibration, manifest |
| `onesystem/evaluate.py` | Exact match, calibration error, temperature fitting |
| `onesystem/cli.py` | `onesystem` command |
| `tests/` | Unit tests; the CI job runs without torch, so tests skip torch-only modules when it is absent |
| `models/onesystem` | Published config, tokenizer, manifest, eval. Weights live in the GitHub release. |

## Publishing weights

Weights are not committed. After training:

```bash
gh release create v0.2.0 models/onesystem/model.safetensors models/onesystem/config.json \
  models/onesystem/tokenizer.json models/onesystem/tokenizer_config.json \
  models/onesystem/special_tokens_map.json models/onesystem/vocab.txt \
  models/onesystem/onesystem.json models/onesystem/eval.json --title "OneSystem v0.2.0"
```

`OneSystem.load("github:owner/repo@tag")` loads any release built this way.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Useful contributions: a new labelled domain, a stronger encoder initialisation, hard-negative label sampling, a multilingual run, or an evaluation on untouched data.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The encoder initialisation (`BAAI/bge-base-en-v1.5`, MIT) and the training data (`fastino/fast-decisions`, Apache 2.0) keep their own licenses.

## Limitations

- The public development split is 1,700 rows across 17 domains. OneSystem is a small specialist; it is not a replacement for a large checkpoint trained on millions of decisions.
- Multi-label heads are scored as exact set match and remain the weakest heads.
- Calibration is fit on the training domains. Check `confidence` against your own labels before branching production traffic on it.
- English only.
