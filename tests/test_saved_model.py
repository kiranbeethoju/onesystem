import json
from pathlib import Path

from onesystem import MODEL_NAME

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "onesystem"


def test_published_config_names_onesystem_and_no_upstream_checkpoint():
    config = json.loads((MODEL_DIR / "config.json").read_text())
    manifest = json.loads((MODEL_DIR / "onesystem.json").read_text())
    assert config["name"] == MODEL_NAME == manifest["name"]
    assert config["model_type"] == "onesystem"
    assert config["encoder_config"]["model_type"]
    assert config["temperature"] > 0
    assert "gliner" not in json.dumps(manifest).lower()
    assert "fastino/GLiNER" not in json.dumps(config)
