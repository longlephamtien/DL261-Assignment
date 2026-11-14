"""Modality A: the vision encoder and the image-only baseline. Implemented by #98."""

from __future__ import annotations

from torch import Tensor, nn

from . import register


class VisionEncoder(nn.Module):
    """DeepLabV3 backbone returning a feature map that keeps spatial layout."""

    def __init__(self, backbone: str = "mobilenet_v3_large", pretrained: bool = True) -> None:
        super().__init__()
        raise NotImplementedError("#98 implements the modality-A encoder")

    def forward(self, images: Tensor) -> Tensor:
        raise NotImplementedError("#98 implements the modality-A encoder")


@register("image-only")
def build_image_only(**kwargs) -> nn.Module:
    """Segments without reading the expression; bounded by the distractor count."""
    raise NotImplementedError("#100 trains the modality-A baseline")
