"""The simple fusion model, checked with a stand-in text encoder so it does not wait on #99."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.interfaces import NUM_CLASSES, PAD_TOKEN
from src.models.fusion import FusionSegmenter, LateConcatFusion, build_simple_fusion, masked_mean
from src.models.vision import VisionEncoder

TEXT_WIDTH = 16


class StubText(nn.Module):
    """Per-token states (N, L, D) from an embedding, the shape the text contract promises."""

    def __init__(self, frozen: bool = True) -> None:
        super().__init__()
        self.embed = nn.Embedding(100, TEXT_WIDTH, padding_idx=PAD_TOKEN)
        self.dropout = nn.Dropout(0.5)
        self.embed.weight.requires_grad_(not frozen)

    def forward(self, tokens, attention):
        return self.dropout(self.embed(tokens))


def inputs(n: int = 2, length: int = 5, size: int = 64):
    tokens = torch.randint(1, 100, (n, length))
    tokens[:, -2:] = PAD_TOKEN
    attention = tokens != PAD_TOKEN
    return torch.randn(n, 3, size, size), tokens, attention


@pytest.fixture(scope="module")
def model() -> FusionSegmenter:
    vision = VisionEncoder(pretrained=False)
    late = LateConcatFusion(VisionEncoder.out_channels, TEXT_WIDTH, width=32)
    return FusionSegmenter(vision, StubText(), late, width=32)


def test_pooling_ignores_padding() -> None:
    states = torch.randn(2, 4, TEXT_WIDTH)
    attention = torch.tensor([[True, True, False, False], [True, True, True, True]])
    changed = states.clone()
    changed[0, 2:] = 1e6
    assert torch.allclose(masked_mean(states, attention), masked_mean(changed, attention))
    assert torch.allclose(masked_mean(states, attention)[0], states[0, :2].mean(0))


def test_an_all_padding_row_does_not_divide_by_zero() -> None:
    pooled = masked_mean(torch.randn(1, 3, TEXT_WIDTH), torch.zeros(1, 3, dtype=torch.bool))
    assert torch.isfinite(pooled).all()


def test_the_fusion_keeps_the_spatial_size_of_the_vision_map() -> None:
    late = LateConcatFusion(960, TEXT_WIDTH, width=32)
    _, tokens, attention = inputs()
    fused = late(torch.randn(2, 960, 20, 20), torch.randn(2, tokens.shape[1], TEXT_WIDTH), attention)
    assert fused.shape == (2, 32, 20, 20)


def test_the_model_returns_two_class_logits_at_the_input_resolution(model) -> None:
    images, tokens, attention = inputs(size=320)
    model.eval()
    with torch.no_grad():
        logits = model(images, tokens, attention)
    assert logits.shape == (2, NUM_CLASSES, 320, 320)


def test_the_expression_changes_the_prediction(model) -> None:
    images, tokens, attention = inputs()
    other = torch.randint(1, 100, tokens.shape).masked_fill(~attention, PAD_TOKEN)
    model.eval()
    with torch.no_grad():
        assert not torch.allclose(model(images, tokens, attention), model(images, other, attention))


def test_a_frozen_text_encoder_stays_in_eval_mode_and_gets_no_gradient(model) -> None:
    model.train()
    assert not model.text.training
    assert model.fusion.training and model.vision.training
    images, tokens, attention = inputs()
    model(images, tokens, attention).sum().backward()
    assert model.text.embed.weight.grad is None
    assert model.fusion.project[0].weight.grad is not None
    model.zero_grad(set_to_none=True)


def test_a_trainable_text_encoder_follows_the_training_mode() -> None:
    vision = VisionEncoder(pretrained=False)
    late = LateConcatFusion(VisionEncoder.out_channels, TEXT_WIDTH, width=32)
    model = FusionSegmenter(vision, StubText(frozen=False), late, width=32).train()
    assert model.text.training


@pytest.mark.parametrize("arguments", [
    {"fusion": "cross-attention"},
    {"modalities": ["image"]},
])
def test_the_builder_rejects_a_config_that_is_not_simple_fusion(arguments) -> None:
    with pytest.raises(ValueError):
        build_simple_fusion(pretrained=False, **arguments)
