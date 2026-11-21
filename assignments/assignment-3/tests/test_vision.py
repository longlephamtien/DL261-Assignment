"""The modality-A encoder: shape, spatial layout and configuration errors."""

from __future__ import annotations

import pytest
import torch

from src.models.vision import VisionEncoder


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
