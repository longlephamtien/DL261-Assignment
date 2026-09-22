"""Shape and capacity contract for every registered model."""

from __future__ import annotations

import pytest
import torch

from src.interfaces import IMAGE_SHAPE, NUM_CLASSES, count_parameters
from src.models import build_model, list_models
from src.utils import load_config, resolve_path

BATCH = 4
LINEAR_PARAMETERS = 784 * NUM_CLASSES + NUM_CLASSES
MLP_PARAMETERS = (784 * 512 + 512) + (512 * 256 + 256) + (256 * NUM_CLASSES + NUM_CLASSES)


@pytest.fixture
def images() -> torch.Tensor:
    return torch.randn(BATCH, *IMAGE_SHAPE)


@pytest.mark.parametrize("name", ["linear", "mlp"])
def test_registered(name: str) -> None:
    assert name in list_models()


@pytest.mark.parametrize("name", ["linear", "mlp"])
def test_returns_logits(name: str, images: torch.Tensor) -> None:
    logits = build_model(name)(images)
    assert logits.shape == (BATCH, NUM_CLASSES)
    assert logits.dtype == torch.float32
    assert not torch.allclose(logits.exp().sum(dim=1), torch.ones(BATCH)), "logits must not be softmax outputs"


def test_linear_parameter_count() -> None:
    assert count_parameters(build_model("linear")) == LINEAR_PARAMETERS


def test_mlp_parameter_count() -> None:
    model = build_model("mlp", hidden_sizes=[512, 256], activation="relu", dropout=0.2)
    assert count_parameters(model) == MLP_PARAMETERS


@pytest.mark.parametrize("name", ["linear", "mlp"])
def test_config_builds(name: str, images: torch.Tensor) -> None:
    config = load_config(resolve_path(f"configs/{name}.yaml"))
    model = build_model(config["model"]["name"], **config["model"].get("args", {}))
    assert model(images).shape == (BATCH, NUM_CLASSES)


def test_mlp_depth_and_width_follow_configuration() -> None:
    narrow = count_parameters(build_model("mlp", hidden_sizes=[64]))
    wide = count_parameters(build_model("mlp", hidden_sizes=[64, 64]))
    assert wide > narrow


def test_mlp_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="unsupported activation"):
        build_model("mlp", activation="sigmoid_swish")


def test_unknown_model_is_reported() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        build_model("resnet")


def test_unknown_argument_is_reported() -> None:
    with pytest.raises(TypeError):
        build_model("linear", hidden_sizes=[512])
