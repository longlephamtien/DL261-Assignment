"""Modality B: the text encoder and the text-only baseline. Implemented by #99."""

from __future__ import annotations

from torch import Tensor, nn

from . import register


class TextEncoder(nn.Module):
    """Pretrained sentence encoder returning per-token states; pooling is the fusion's choice."""

    def __init__(self, name: str = "distilbert-base-uncased", frozen: bool = True) -> None:
        super().__init__()
        raise NotImplementedError("#99 implements the modality-B encoder")

    def forward(self, tokens: Tensor, attention: Tensor) -> Tensor:
        raise NotImplementedError("#99 implements the modality-B encoder")


@register("text-only")
def build_text_only(**kwargs) -> nn.Module:
    """Predicts a mask from the expression alone, as a spatial prior. Weak by construction."""
    raise NotImplementedError("#101 trains the modality-B baseline")
