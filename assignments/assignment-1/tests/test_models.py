"""Shape and capacity contract for every registered model."""

from __future__ import annotations

import pytest
import torch

from src.interfaces import IMAGE_SHAPE, NUM_CLASSES, count_parameters
from src.models import build_model, list_models
from src.data import token_shape
from src.utils import load_config, resolve_path

BATCH = 4
LINEAR_PARAMETERS = 784 * NUM_CLASSES + NUM_CLASSES
MLP_PARAMETERS = (784 * 512 + 512) + (512 * 256 + 256) + (256 * NUM_CLASSES + NUM_CLASSES)
ALL_MODELS = ["linear", "mlp", "transformer", "recurrent"]


def transformer_parameters(tokens: int, features: int, d_model: int, ff_dim: int, layers: int) -> int:
    """Projection, class token, positions, encoder layers, final norm, and head."""
    attention = 3 * d_model * d_model + 3 * d_model + d_model * d_model + d_model
    feedforward = (d_model * ff_dim + ff_dim) + (ff_dim * d_model + d_model)
    layer = attention + feedforward + 4 * d_model
    return (
        (features * d_model + d_model)
        + d_model
        + (tokens + 1) * d_model
        + layers * layer
        + 2 * d_model
        + (d_model * NUM_CLASSES + NUM_CLASSES)
    )


@pytest.fixture
def images() -> torch.Tensor:
    return torch.randn(BATCH, *IMAGE_SHAPE)


@pytest.mark.parametrize("name", ALL_MODELS)
def test_registered(name: str) -> None:
    assert name in list_models()


@pytest.mark.parametrize("name", ALL_MODELS)
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
@pytest.mark.parametrize("shape", [(BATCH, *IMAGE_SHAPE), (BATCH, 784), (784,)])
def test_accepts_image_flat_and_unbatched_input(name: str, shape: tuple[int, ...]) -> None:
    logits = build_model(name)(torch.randn(*shape))
    assert logits.shape == ((NUM_CLASSES,) if len(shape) == 1 else (BATCH, NUM_CLASSES))


@pytest.mark.parametrize("name", ALL_MODELS)
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


# --- Recurrent Tests ---

@pytest.mark.parametrize("cell_type", ["lstm", "gru"])
@pytest.mark.parametrize("representation", ["rows", "columns", "patches"])
def test_recurrent_output_shape(cell_type: str, representation: str, images: torch.Tensor) -> None:
    model = build_model(
        "recurrent",
        cell_type=cell_type,
        representation=representation,
        patch_size=4,
        hidden_size=64,
        num_layers=1,
    )
    logits = model(images)
    assert logits.shape == (BATCH, NUM_CLASSES)
    assert logits.dtype == torch.float32


@pytest.mark.parametrize("cell_type", ["lstm", "gru"])
def test_recurrent_parameter_count(cell_type: str) -> None:
    model = build_model("recurrent", cell_type=cell_type, representation="rows", hidden_size=128)
    params = count_parameters(model)
    assert params > 0

# --- Transformer Tests ---

@pytest.mark.parametrize(
    ("representation", "tokens", "features"),
    [("rows", 28, 28), ("columns", 28, 28), ("patches", 49, 16)],
)
def test_transformer_tokenizes_by_representation(
    representation: str, tokens: int, features: int, images: torch.Tensor
) -> None:
    assert token_shape(representation, patch_size=4) == (tokens, features)
    assert build_model("transformer", representation=representation)(images).shape == (BATCH, NUM_CLASSES)


def test_transformer_parameter_count() -> None:
    model = build_model("transformer", representation="patches", patch_size=4)
    assert count_parameters(model) == transformer_parameters(49, 16, d_model=128, ff_dim=256, layers=4)


def test_transformer_pooling_changes_the_readout() -> None:
    cls = build_model("transformer", pooling="cls")
    mean = build_model("transformer", pooling="mean")
    assert cls.class_token is not None and mean.class_token is None
    assert count_parameters(cls) - count_parameters(mean) == 2 * 128, "class token plus its position"


def test_transformer_rejects_the_image_representation() -> None:
    with pytest.raises(ValueError, match="not a sequence"):
        build_model("transformer", representation="image")


def test_transformer_rejects_unsupported_pooling() -> None:
    with pytest.raises(ValueError, match="unsupported pooling"):
        build_model("transformer", pooling="attention")


def test_transformer_rejects_indivisible_patch_size() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        build_model("transformer", representation="patches", patch_size=5)


def test_transformer_rejects_heads_that_do_not_divide_the_width() -> None:
    with pytest.raises(ValueError, match="divisible by num_heads"):
        build_model("transformer", d_model=128, num_heads=5)


def test_transformer_requires_batched_images() -> None:
    with pytest.raises(ValueError, match=r"\(N, 1, 28, 28\)"):
        build_model("transformer")(torch.randn(784))


@pytest.mark.parametrize("model", ["transformer", "recurrent"])
def test_sequence_config_agrees_with_the_input_block(model: str) -> None:
    """`input` labels the run and `model.args` drives the forward pass; a mismatch misreports it."""
    config = load_config(resolve_path(f"configs/{model}.yaml"))
    args = config["model"]["args"]
    assert args["representation"] == config["input"]["representation"]
    assert args["patch_size"] == config["input"]["patch_size"]


@pytest.mark.parametrize("model", ["linear", "mlp", "transformer", "recurrent"])
def test_config_leaves_the_shared_training_protocol_alone(model: str) -> None:
    """Every model trains under one protocol; tuning varies `model.args` only."""
    base = load_config(resolve_path("configs/base.yaml"))["train"]
    assert load_config(resolve_path(f"configs/{model}.yaml"))["train"] == base


def available_devices() -> list[torch.device]:
    """CPU always, plus whichever accelerator this machine offers."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


@pytest.mark.parametrize("name", ALL_MODELS)
def test_runs_on_every_available_device(name: str, images: torch.Tensor) -> None:
    for device in available_devices():
        model = build_model(name).to(device)
        assert {p.device.type for p in model.parameters()} == {device.type}
        logits = model(images.to(device))
        assert logits.shape == (BATCH, NUM_CLASSES)
        assert logits.device.type == device.type


def test_transformer_matches_across_devices(images: torch.Tensor) -> None:
    """Without dropout the maths must agree; with it, each device draws its own masks."""
    devices = available_devices()
    if len(devices) < 2:
        pytest.skip("no accelerator to compare against")

    reference = build_model("transformer", dropout=0.0)
    state = reference.state_dict()
    outputs = []
    for device in devices:
        model = build_model("transformer", dropout=0.0).to(device)
        model.load_state_dict({key: value.to(device) for key, value in state.items()})
        model.eval()
        with torch.no_grad():
            outputs.append(model(images.to(device)).cpu())
    assert torch.allclose(outputs[0], outputs[1], atol=1e-4)
