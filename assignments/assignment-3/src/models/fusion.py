"""Fusion of the two streams. Simple fusion by #102, cross-attention by #104."""

from __future__ import annotations

from torch import Tensor, nn

from . import register


class LateConcatFusion(nn.Module):
    """One pooled sentence vector broadcast onto the feature map and concatenated."""

    def __init__(self, vision_width: int, text_width: int, width: int = 256) -> None:
        super().__init__()
        raise NotImplementedError("#102 implements the simple fusion baseline")

    def forward(self, vision: Tensor, text: Tensor, attention: Tensor) -> Tensor:
        raise NotImplementedError("#102 implements the simple fusion baseline")


class CrossAttentionFusion(nn.Module):
    """Each visual position attends over the tokens, so a word can address a region."""

    def __init__(self, vision_width: int, text_width: int, heads: int = 8) -> None:
        super().__init__()
        raise NotImplementedError("#104 implements the improved fusion model")

    def forward(self, vision: Tensor, text: Tensor, attention: Tensor) -> Tensor:
        raise NotImplementedError("#104 implements the improved fusion model")


@register("simple-fusion")
def build_simple_fusion(**kwargs) -> nn.Module:
    raise NotImplementedError("#102 implements the simple fusion baseline")


@register("improved-fusion")
def build_improved_fusion(**kwargs) -> nn.Module:
    raise NotImplementedError("#104 implements the improved fusion model")
