"""The modality-A encoder: shape, spatial layout and configuration errors."""

from __future__ import annotations

import pytest
import torch

from src.interfaces import NUM_CLASSES
from src.models.vision import ImageOnly, VisionEncoder, build_image_only


@pytest.fixture(scope="module")
def encoder() -> VisionEncoder:
    # Random weights: the shape contract does not depend on the checkpoint, and the test must
    # not download one.
    return VisionEncoder(pretrained=False).eval()


def test_a_320_square_comes_out_as_a_20_by_20_map(encoder: VisionEncoder) -> None:
    with torch.no_grad():
        features = encoder(torch.randn(2, 3, 320, 320))
    assert features.shape == (2, VisionEncoder.out_channels, 20, 20)


def test_the_map_follows_the_input_at_stride_16(encoder: VisionEncoder) -> None:
    with torch.no_grad():
        features = encoder(torch.randn(1, 3, 256, 384))
    assert features.shape[-2:] == (256 // VisionEncoder.stride, 384 // VisionEncoder.stride)


def test_the_auxiliary_head_is_not_kept(encoder: VisionEncoder) -> None:
    names = {name for name, _ in encoder.named_modules()}
    assert not any("aux" in name or "classifier" in name for name in names)


def test_an_unknown_backbone_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="unknown backbone"):
        VisionEncoder(backbone="resnet50", pretrained=False)


def test_the_image_only_baseline_returns_two_class_logits_at_the_input_size() -> None:
    model = ImageOnly(pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 320, 320), None, None)
    assert logits.shape == (2, NUM_CLASSES, 320, 320)
    assert model.network.aux_classifier is None


def test_the_image_only_baseline_ignores_the_expression() -> None:
    model = ImageOnly(pretrained=False).eval()
    images = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        first = model(images, torch.tensor([[1, 2]]), torch.ones(1, 2, dtype=torch.bool))
        second = model(images, torch.tensor([[3, 4]]), torch.ones(1, 2, dtype=torch.bool))
    assert torch.equal(first, second)


def test_the_image_only_builder_refuses_a_second_modality() -> None:
    with pytest.raises(ValueError, match="image alone"):
        build_image_only(modalities=["image", "text"], pretrained=False)
