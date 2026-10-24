"""DeepLabV3 with a pretrained torchvision backbone, re-headed for the 19 Cityscapes classes.

The published weights were trained on COCO with VOC labels, so they predict 21 classes. The
backbone and the ASPP module transfer; only the final 1x1 convolution is replaced, which is the
one layer whose shape depends on the label set. Implemented by #64.
"""

from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import segmentation as tv

from ..interfaces import NUM_CLASSES
from . import register

BACKBONES = {
    "mobilenet_v3_large": (tv.deeplabv3_mobilenet_v3_large, tv.DeepLabV3_MobileNet_V3_Large_Weights),
    "resnet50": (tv.deeplabv3_resnet50, tv.DeepLabV3_ResNet50_Weights),
    "resnet101": (tv.deeplabv3_resnet101, tv.DeepLabV3_ResNet101_Weights),
}

def normalization_for(weights) -> dict:
    transforms = weights.transforms()
    return {
        "mean": list(transforms.mean),
        "std": list(transforms.std),
        "source": weights.name,
    }


class DeepLabV3Segmenter(nn.Module):
    """Wraps the torchvision model so the registry contract holds: images in, logits out."""

    def __init__(
        self,
        backbone: str = "mobilenet_v3_large",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unsupported backbone '{backbone}'; expected one of {', '.join(BACKBONES)}")

        builder, weights_enum = BACKBONES[backbone]
        weights = weights_enum.DEFAULT if pretrained else None

        self.network = builder(weights=weights, aux_loss=True)
        # The head width is whatever ASPP emits for this backbone, so read it off the layer
        # being replaced rather than assuming a value that only holds for some backbones.
        head = self.network.classifier[-1]
        self.network.classifier[-1] = nn.Conv2d(head.in_channels, num_classes, kernel_size=1)
        self.network.aux_classifier = None

        self.backbone_name = backbone
        self.pretrained = pretrained
        self.weights_source = weights.name if weights is not None else "random initialisation"
        self.weights_url = weights.url if weights is not None else ""
        self.normalization = normalization_for(weights) if weights is not None else None

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Hold the pretrained feature extractor fixed and train the head alone."""
        for parameter in self.network.backbone.parameters():
            parameter.requires_grad = False

    def forward(self, images: Tensor) -> Tensor:
        return self.network(images)["out"]


@register("deeplabv3")
def build_deeplabv3(**kwargs) -> nn.Module:
    return DeepLabV3Segmenter(**kwargs)
