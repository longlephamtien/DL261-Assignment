"""Modality A: the vision encoder and the image-only baseline. Implemented by #98."""

from __future__ import annotations

from torch import Tensor, nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from . import register

BACKBONES = ("mobilenet_v3_large",)


class VisionEncoder(nn.Module):
    """DeepLabV3 backbone returning a feature map that keeps spatial layout.

    The MobileNetV3-Large trunk runs at output stride 16, its last stage dilated rather than
    strided, so a 320 square comes out as a (N, 960, 20, 20) map: one feature vector per 16-pixel
    cell, which a fusion module can address by position.
    """

    out_channels = 960
    stride = 16

    def __init__(self, backbone: str = "mobilenet_v3_large", pretrained: bool = True) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; expected one of {BACKBONES}")
        # The published COCO weights were fitted with the auxiliary head, so the network must be
        # built with it to load them. Only the trunk is kept: the fusion supplies its own head.
        network = deeplabv3_mobilenet_v3_large(
            weights="DEFAULT" if pretrained else None, weights_backbone=None, aux_loss=True
        )
        self.backbone = network.backbone

    def forward(self, images: Tensor) -> Tensor:
        return self.backbone(images)["out"]


@register("image-only")
def build_image_only(**kwargs) -> nn.Module:
    """Segments without reading the expression; bounded by the distractor count."""
    raise NotImplementedError("#100 trains the modality-A baseline")
