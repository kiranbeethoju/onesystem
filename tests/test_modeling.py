import pytest

torch = pytest.importorskip("torch")

from onesystem.modeling import (  # noqa: E402
    OneSystemConfig,
    _identity_projection,
    expected_level,
    labels_look_ordinal,
    masked_mean,
)


def test_projection_starts_as_identity():
    layer = _identity_projection(8)
    x = torch.randn(3, 8)
    assert torch.allclose(layer(x), x, atol=1e-6)


def test_masked_mean_ignores_padding():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(masked_mean(hidden, mask), torch.tensor([[2.0, 2.0]]))


def test_ordinal_helpers():
    assert labels_look_ordinal(["0", "1", "2", "3"])
    assert not labels_look_ordinal(["low", "high"])
    assert expected_level([0.0, 0.0, 1.0]) == 2.0
    assert expected_level([0.5, 0.5]) == 0.5


def test_config_round_trip(tmp_path):
    config = OneSystemConfig(encoder_name="x", encoder_config={"model_type": "bert"}, temperature=1.7)
    config.save(tmp_path)
    loaded = OneSystemConfig.load(tmp_path)
    assert loaded.temperature == 1.7
    assert loaded.encoder_config == {"model_type": "bert"}
    assert loaded.name == "OneSystem"
