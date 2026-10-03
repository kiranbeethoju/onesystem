from onesystem import BASE_CHECKPOINT, DATASET_ID, MODEL_NAME, __version__
from onesystem.evaluate import gold_labels, predicted_labels, tasks_from_record
from onesystem.train import build_manifest, build_training_config


def test_project_name_is_onesystem():
    assert MODEL_NAME == "OneSystem"
    assert __version__ == "0.1.0"
    assert BASE_CHECKPOINT == "fastino/GLiNER2.5-Decide"
    assert DATASET_ID == "fastino/fast-decisions"


def test_training_config_is_a_small_lora_run():
    config = build_training_config()
    assert config.experiment_name == "OneSystem"
    assert config.use_lora is True
    assert config.save_adapter_only is True
    assert config.fp16 is False
    assert config.bf16 is False
    assert config.num_epochs == 1
    assert config.batch_size == 1
    assert "encoder" in config.lora_target_modules


def test_manifest_names_the_saved_model():
    document = build_manifest({"domains": 17, "train_rows": 1360, "eval_rows": 340})
    assert document["name"] == "OneSystem"
    assert document["base_checkpoint"] == BASE_CHECKPOINT
    assert document["dataset_split"] == "development"
    assert document["train_rows"] == 1360


def test_prediction_helpers_compare_sets():
    record = {
        "input": "Stop the bot and get me a person.",
        "output": {
            "classifications": [
                {
                    "task": "handoff",
                    "labels": ["yes", "no"],
                    "true_label": ["yes"],
                    "multi_label": False,
                }
            ]
        },
    }
    assert tasks_from_record(record)["handoff"]["labels"] == ["yes", "no"]
    assert gold_labels(record["output"]["classifications"][0]) == ["yes"]
    assert predicted_labels({"label": "yes", "confidence": 0.8}) == ["yes"]
    assert predicted_labels(["billing", "access"]) == ["billing", "access"]
    assert predicted_labels(
        [{"label": "checkout", "confidence": 0.7}, {"label": "notifications", "confidence": 0.9}]
    ) == ["checkout", "notifications"]
