# Contributing to OneSystem

OneSystem is an open-source specialist decision model. Contributions are welcome from people who want better operational decisions: routing, document type, handoff, and the domains already in the public development data.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[local,dev]"
python -m pytest
```

The unit tests do not download model weights. Training and evaluation do.

## What to contribute

- A bug fix, with a test that fails before the fix.
- A labelled domain in the existing jsonl shape: `input` plus `output.classifications`, each head carrying `task`, `labels`, `true_label`, and `multi_label`.
- Evaluation on data that was not used for training. Say where the data came from and keep a held-out slice.
- Documentation that matches the code.

## Data rules

The files in `fastino/fast-decisions` are development examples. Keep a holdout. Do not describe a score on those files, or on a reshuffle of them, as Fastino's published benchmark. If you add data, you need the right to release it under Apache 2.0, and the gold labels must be members of the candidate set.

## Pull requests

1. Fork the repository and create a branch.
2. Keep the change focused.
3. Run `python -m pytest`.
4. Describe what changed and how you checked it.
5. Do not commit virtual environments, raw downloads, secrets, or the Hugging Face base weights.

## Saving a model

`python -m onesystem.train` writes the adapter to `models/onesystem` and a manifest named OneSystem. If you change the training recipe, update `onesystem.json` fields in the manifest writer and the README so the saved name still describes this run.

## Conduct

Be respectful in issues and reviews. Critique the code and the experiment, and keep personal information out of examples you commit.
