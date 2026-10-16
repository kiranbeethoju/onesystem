# Offline / air-gapped bundle checklist

**Ready-made pack:** [offline_bundle.zip](https://github.com/kiranbeethoju/onesystem/releases/download/v0.3.0/offline_bundle.zip) on the [v0.3.0 release](https://github.com/kiranbeethoju/onesystem/releases/tag/v0.3.0) — full source + `models/onesystem/` weights (~480 MB). Unzip, then install deps from a local `wheelhouse/` (see below) or any already-available torch/transformers.

Use this when a Linux machine has **no internet** and you cannot `pip install` from PyPI or Hugging Face.

## On a connected machine

```bash
# 1) Source
git clone --depth 1 --branch v0.3.0 https://github.com/kiranbeethoju/onesystem.git
cd onesystem

# 2) Release weights into models/onesystem/ (required files)
mkdir -p models/onesystem/encoder
BASE=https://github.com/kiranbeethoju/onesystem/releases/download/v0.3.0
for f in model.safetensors config.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json vocab.txt onesystem.json eval.json \
         encoder-config.json encoder-configuration.py encoder-modeling.py; do
  curl -L -o "/tmp/$f" "$BASE/$f"
done
mv /tmp/model.safetensors /tmp/config.json /tmp/tokenizer.json \
   /tmp/tokenizer_config.json /tmp/special_tokens_map.json /tmp/vocab.txt \
   /tmp/onesystem.json /tmp/eval.json models/onesystem/
mv /tmp/encoder-config.json models/onesystem/encoder/config.json
mv /tmp/encoder-configuration.py models/onesystem/encoder/configuration.py
mv /tmp/encoder-modeling.py models/onesystem/encoder/modeling.py

# 3) Dependency wheels (match the isolated machine's Python + OS + CPU/GPU)
python3 -m venv /tmp/os-venv && source /tmp/os-venv/bin/activate
pip install -U pip
mkdir -p wheelhouse
pip download -r requirements.txt -d wheelhouse/

# 4) Pack
cd ..
tar -czf onesystem-offline.tgz onesystem/
```

Copy `onesystem-offline.tgz` to the isolated host (USB, scp bastion, etc.).

## On the isolated Linux machine

```bash
tar -xzf onesystem-offline.tgz && cd onesystem
python3 -m venv .venv && source .venv/bin/activate
pip install --no-index --find-links=wheelhouse -r requirements.txt

# Option A — editable install (still local, no network)
pip install --no-index --find-links=wheelhouse -e .

# Option B — no pip install of onesystem at all
export PYTHONPATH="$PWD:$PYTHONPATH"

# Always point at local weights (never call OneSystem.load() without a path
# unless models/onesystem/model.safetensors already exists — that would try GitHub).
python - <<'PY'
from onesystem.model import OneSystem
model = OneSystem.load("models/onesystem")  # local only
print(model.classify("Freeze my card now.", {"intent": ["freeze_card", "other"]}))
PY

# Recipe smoke test
python examples/run_recipes.py --model models/onesystem --out /tmp/recipes.json
```

## Rules for air-gap

| Do | Don't |
|---|---|
| `OneSystem.load("models/onesystem")` | `OneSystem.load()` if weights are missing (downloads release) |
| Ship `wheelhouse/` + `requirements.txt` | `pip install onesystem` / HF downloads at runtime |
| Set `PYTHONPATH=repo_root` if skipping `pip install -e .` | Expect transformers to fetch encoder code from Hub |
| Keep `models/onesystem/encoder/*.py` with the weights | Delete encoder/ architecture files |

Optional: `export ONESYSTEM_HOME=/path/to/cache` if you store weights outside the repo (only used when resolving the default cache path after a prior download).
