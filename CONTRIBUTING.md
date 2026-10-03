# Contributing to OneSystem

OneSystem is an open-source System 1 decision model with its own weights and a training loop you can read in one sitting. Contributions are welcome from people who want better operational decisions: routing, document type, handoff, urgency, and the domains already in the public development data.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m flake8 . --select=E9,F63,F7,F82 --exclude .venv
python -m pytest
```

The unit tests do not download weights. CI installs only `flake8` and `pytest`, so any test that needs torch must start with `pytest.importorskip("torch")`.

## What to contribute

- A bug fix, with a test that fails before the fix.
- A labelled domain in the existing jsonl shape: `input` plus `output.classifications`, each head carrying `task`, `labels`, `true_label`, and `multi_label`.
- A training improvement: hard-negative labels, a different encoder initialisation (`--encoder`), longer context, a multilingual run. Include the `onesystem.json` manifest and `eval.json` from your run.
- Evaluation on data that was not used for training. Say where the data came from and keep a held-out slice.
- Documentation that matches the code.

## Data rules

The files in `fastino/fast-decisions` are development examples. Keep the eval slice untouched and fit calibration only on the calibration slice. Do not describe a score on those files, or on a reshuffle of them, as Fastino's published benchmark. If you add data, you need the right to release it under Apache 2.0, and gold labels must be members of the candidate set.

## Pull requests

1. Fork the repository and create a branch.
2. Keep the change focused.
3. Run flake8 and pytest as above.
4. Describe what changed and how you checked it.
5. Do not commit virtual environments, raw downloads, secrets, or `model.safetensors`. Weights go in a GitHub release.

## Releasing weights

`python -m onesystem.train` writes the full model to `models/onesystem`. Commit `config.json`, the tokenizer files, `onesystem.json`, `eval.json`, and the model README; publish `model.safetensors` and the same files as release assets under the tag named in `onesystem/__init__.py` (`RELEASE_TAG`). Bump `__version__`, `RELEASE_TAG`, and `pyproject.toml` together.

## Conduct

Be respectful in issues and reviews. Critique the code and the experiment, and keep personal information out of examples you commit.
