"""Modality A: the vision encoder and the image-only baseline. Implemented by #98."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from ..interfaces import NUM_CLASSES
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


class ImageOnly(nn.Module):
    """Modality A alone: the whole DeepLabV3, re-headed for one foreground class.

    It keeps the COCO-pretrained ASPP head rather than the fusion models' mask head, because a
    baseline should be the strongest image model the weights allow, not a fusion model with the
    text removed; that second comparison is the E4 ablation.
    """

    def __init__(self, backbone: str = "mobilenet_v3_large", pretrained: bool = True) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; expected one of {BACKBONES}")
        self.network = deeplabv3_mobilenet_v3_large(
            weights="DEFAULT" if pretrained else None, weights_backbone=None, aux_loss=True
        )
        # Built with the auxiliary head so the published weights load, then dropped: no loss reads it.
        self.network.aux_classifier = None
        last = self.network.classifier[-1]
        self.network.classifier[-1] = nn.Conv2d(last.in_channels, NUM_CLASSES, 1)

    def forward(self, images: Tensor, tokens: Tensor | None = None,
                attention: Tensor | None = None) -> Tensor:
        # torchvision already upsamples the logits to the input size.
        return self.network(images)["out"]


@register("image-only")
def build_image_only(
    modalities: Sequence[str] = ("image",),
    backbone: str = "mobilenet_v3_large",
    pretrained: bool = True,
) -> nn.Module:
    """Segments without reading the expression; bounded by the distractor count."""
    if list(modalities) != ["image"]:
        raise ValueError(f"the image-only baseline reads the image alone, got {list(modalities)}")
    return ImageOnly(backbone, pretrained)
