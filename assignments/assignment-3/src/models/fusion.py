"""Fusion of the two streams. Simple fusion by #102, cross-attention by #104."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..interfaces import NUM_CLASSES
from . import register
from .text import TextEncoder
from .vision import VisionEncoder


def masked_mean(states: Tensor, attention: Tensor) -> Tensor:
    """Average the real tokens of (N, L, D) states into (N, D); padding contributes nothing."""
    weights = attention.unsqueeze(-1).to(states.dtype)
    return (states * weights).sum(1) / weights.sum(1).clamp(min=1)


class LateConcatFusion(nn.Module):
    """One pooled sentence vector broadcast onto the feature map and concatenated."""

    def __init__(self, vision_width: int, text_width: int, width: int = 256) -> None:
        super().__init__()
        self.out_channels = width
        self.project = nn.Sequential(
            nn.Conv2d(vision_width + text_width, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

    def forward(self, vision: Tensor, text: Tensor, attention: Tensor) -> Tensor:
        sentence = masked_mean(text, attention)
        spread = sentence[:, :, None, None].expand(-1, -1, *vision.shape[-2:])
        return self.project(torch.cat([vision, spread.to(vision.dtype)], dim=1))


class CrossAttentionFusion(nn.Module):
    """Each visual position attends over the tokens, so a word can address a region."""

    def __init__(self, vision_width: int, text_width: int, heads: int = 8) -> None:
        super().__init__()
        raise NotImplementedError("#104 implements the improved fusion model")

    def forward(self, vision: Tensor, text: Tensor, attention: Tensor) -> Tensor:
        raise NotImplementedError("#104 implements the improved fusion model")


class FusionSegmenter(nn.Module):
    """Vision encoder, text encoder and a fusion, read by one mask head.

    The fusion returns a map at the encoder's 1/16 resolution; the head turns it into two-class
    logits, upsampled bilinearly to the input, so every fusion model shares one decoder and the
    ablation varies only the fusion.
    """

    def __init__(self, vision: nn.Module, text: nn.Module, fusion: nn.Module, width: int) -> None:
        super().__init__()
        self.vision = vision
        self.text = text
        self.fusion = fusion
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, NUM_CLASSES, 1),
        )

    @property
    def text_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.text.parameters())

    def train(self, mode: bool = True) -> FusionSegmenter:
        super().train(mode)
        # A frozen encoder must not run dropout while the rest trains: its states would be noisy
        # for nothing, since no gradient reaches it.
        if self.text_frozen:
            self.text.eval()
        return self

    def forward(self, images: Tensor, tokens: Tensor, attention: Tensor) -> Tensor:
        visual = self.vision(images)
        states = self.text(tokens, attention)
        logits = self.head(self.fusion(visual, states, attention))
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


def text_width(name: str) -> int:
    """Hidden size of a Hugging Face checkpoint, read from its config without the weights."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(name).hidden_size


def check_arguments(modalities: Sequence[str], fusion: str, expected: str) -> None:
    if sorted(modalities) != ["image", "text"]:
        raise ValueError(f"a fusion model reads both modalities, got {list(modalities)}")
    if fusion != expected:
        raise ValueError(f"this model fuses by {expected!r}, got {fusion!r}")


@register("simple-fusion")
def build_simple_fusion(
    modalities: Sequence[str] = ("image", "text"),
    fusion: str = "late-concat",
    backbone: str = "mobilenet_v3_large",
    encoder: str = "distilbert-base-uncased",
    frozen_text: bool = True,
    width: int = 256,
    pretrained: bool = True,
) -> nn.Module:
    """Late fusion over a frozen text encoder: model 3 of the proposal."""
    check_arguments(modalities, fusion, "late-concat")
    vision = VisionEncoder(backbone=backbone, pretrained=pretrained)
    text = TextEncoder(name=encoder, frozen=frozen_text)
    late = LateConcatFusion(VisionEncoder.out_channels, text_width(encoder), width)
    return FusionSegmenter(vision, text, late, width)


@register("improved-fusion")
def build_improved_fusion(**kwargs) -> nn.Module:
    raise NotImplementedError("#104 implements the improved fusion model")
