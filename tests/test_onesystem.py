import pytest

from onesystem import DATASET_ID, MODEL_NAME, RELEASE_REPO, RELEASE_TAG, __version__
from onesystem.evaluate import expected_calibration_error, gold_labels, predicted_labels, tasks_from_record
from onesystem.hub import REQUIRED_FILES, release_url
from onesystem.model import normalise_tasks


def test_project_identity():
    assert MODEL_NAME == "OneSystem"
    assert __version__ == "0.2.1"
    assert DATASET_ID == "fastino/fast-decisions"
    assert RELEASE_REPO == "kiranbeethoju/onesystem"


def test_release_url_points_at_github_assets():
    url = release_url("model.safetensors")
    assert url == f"https://github.com/kiranbeethoju/onesystem/releases/download/{RELEASE_TAG}/model.safetensors"
    assert "model.safetensors" in REQUIRED_FILES and "config.json" in REQUIRED_FILES


def test_normalise_tasks_accepts_lists_and_dicts():
    specs = normalise_tasks({
        "intent": ["refund", "other"],
        "areas": {"labels": ["billing", "mobile"], "multi_label": True, "threshold": 0.4},
        "card": {"labels": {"pin": "new PIN", "lost": "card missing"}},
    })
    assert specs["intent"]["labels"] == ["refund", "other"]
    assert specs["intent"]["multi_label"] is False
    assert specs["areas"]["multi_label"] is True and specs["areas"]["threshold"] == 0.4
    assert specs["card"]["labels"] == ["pin", "lost"]
    assert specs["card"]["descriptions"]["pin"] == "new PIN"


def test_normalise_tasks_rejects_bad_input():
    with pytest.raises(ValueError):
        normalise_tasks({})
    with pytest.raises(ValueError, match="at least two"):
        normalise_tasks({"x": ["only"]})
    with pytest.raises(ValueError, match="duplicate"):
        normalise_tasks({"x": ["a", "a"]})


def test_prediction_helpers_compare_sets():
    record = {
        "input": "Stop the bot and get me a person.",
        "output": {"classifications": [
            {"task": "handoff", "labels": ["yes", "no"], "true_label": ["yes"], "multi_label": False},
        ]},
    }
    assert tasks_from_record(record)["handoff"]["labels"] == ["yes", "no"]
    assert gold_labels(record["output"]["classifications"][0]) == ["yes"]
    assert predicted_labels({"label": "yes", "confidence": 0.8}) == ["yes"]
    assert predicted_labels({"labels": ["checkout", "notifications"]}) == ["checkout", "notifications"]
    assert predicted_labels({"label": None, "abstain": True}) == []


def test_expected_calibration_error_is_zero_when_confidence_matches_accuracy():
    assert expected_calibration_error([1.0, 1.0], [1, 1]) == 0.0
    assert expected_calibration_error([0.95, 0.95], [0, 0]) == pytest.approx(0.95)
