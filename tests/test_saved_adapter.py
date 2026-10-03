import json
from pathlib import Path

from onesystem import MODEL_NAME

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "models" / "onesystem"


def test_saved_adapter_is_named_onesystem():
    manifest = json.loads((ADAPTER / "onesystem.json").read_text())
    config = json.loads((ADAPTER / "adapter_config.json").read_text())
    assert manifest["name"] == MODEL_NAME == "OneSystem"
    assert config["base_model_name_or_path"] == "fastino/GLiNER2.5-Decide"
    assert config["peft_type"] == "LORA"
    assert (ADAPTER / "adapter_model.safetensors").stat().st_size > 0
