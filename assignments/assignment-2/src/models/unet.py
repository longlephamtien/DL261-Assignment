"""U-Net: the simple baseline, trained from scratch.

The skip connections are the point of the architecture. Pooling four times reduces a 512-pixel
crop to 32 pixels, which is enough context to say what a region is but far too coarse to say
where its boundary runs. Each decoder stage therefore concatenates the encoder map of the same
resolution, so the upsampled semantics are placed back onto the detail that produced them.
Implemented by #65.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..interfaces import NUM_CLASSES
from . import register


def double_conv(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
    """Two padded 3x3 convolutions, each normalised and rectified, at one resolution.

    Padding keeps the spatial size, so only pooling and transposed convolution change it and
    every skip connection meets a map of exactly its own resolution.
    """
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    """Encoder, bottleneck and decoder, with one skip connection per resolution.

    At the defaults, a (N, 3, 512, 512) batch moves through widths 32, 64, 128, 256 while the
    resolution halves to 32, is bridged at width 512, and is mirrored back to 512 pixels before
    a 1x1 convolution maps the 32 channels to the 19 class logits.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_width: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, not {depth}")
        if base_width < 1:
            raise ValueError(f"base_width must be at least 1, not {base_width}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), not {dropout}")

        widths = [base_width * 2**level for level in range(depth)]

        self.encoders = nn.ModuleList()
        channels = in_channels
        for width in widths:
            self.encoders.append(double_conv(channels, width, dropout))
            channels = width

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(channels, channels * 2, dropout)

        # Each stage halves the channel count, then the skip connection doubles it again before
        # the convolutions, which is why the block input is twice its output.
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        channels *= 2
        for width in reversed(widths):
            self.upsamples.append(nn.ConvTranspose2d(channels, width, kernel_size=2, stride=2))
            self.decoders.append(double_conv(width * 2, width, dropout))
            channels = width

        self.head = nn.Conv2d(base_width, num_classes, kernel_size=1)
        self.depth = depth

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[-1] % 2**self.depth or images.shape[-2] % 2**self.depth:
            raise ValueError(
                f"input {tuple(images.shape[-2:])} is not divisible by {2**self.depth}; "
                "pooling would lose a row or column the decoder cannot restore"
            )

        skips: list[Tensor] = []
        features = images
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
            features = self.pool(features)

        features = self.bottleneck(features)

        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            features = upsample(features)
            features = decoder(torch.cat([skip, features], dim=1))

        return self.head(features)


@register("unet")
def build_unet(**kwargs) -> nn.Module:
    return UNet(**kwargs)
