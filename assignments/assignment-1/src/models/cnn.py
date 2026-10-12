"""Convolutional classifier: learned local filters over the image grid, then a fully connected head."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from ..interfaces import IMAGE_SHAPE, NUM_CLASSES
from . import register

KERNEL_SIZE = 3
POOL_SIZE = 2


class ConvClassifier(nn.Module):
    """Stages of convolution, ReLU, and max pooling, then a hidden layer and the logits.

    For a batch of N images with the default channels (32, 64):

        stage 1     (N, 1, 28, 28)  -> conv -> (N, 32, 28, 28) -> pool -> (N, 32, 14, 14)
        stage 2     (N, 32, 14, 14) -> conv -> (N, 64, 14, 14) -> pool -> (N, 64, 7, 7)
        flatten     (N, 64, 7, 7)   -> (N, 3136)
        head        (N, 3136)       -> (N, hidden_size) -> (N, 10)

    Each output channel is one filter of shape (in_channels, 3, 3), applied at every
    position, so its feature map records where that filter responds. Padding keeps the
    convolution at its input size; only pooling shrinks the map, halving it per stage.
    """

    def __init__(
        self,
        channels: Sequence[int] = (32, 64),
        hidden_size: int = 128,
        dropout: float = 0.0,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        in_channels, size, _ = IMAGE_SHAPE
        if not channels:
            raise ValueError("a CNN needs at least one convolutional stage")
        if size // POOL_SIZE ** len(channels) == 0:
            raise ValueError(f"{len(channels)} pooling stages reduce a {size}x{size} image below 1x1")

        layers: list[nn.Module] = []
        for out_channels in channels:
            layers += [
                nn.Conv2d(in_channels, out_channels, KERNEL_SIZE, padding=KERNEL_SIZE // 2),
                nn.ReLU(),
                nn.MaxPool2d(POOL_SIZE),
            ]
            in_channels = out_channels
            size //= POOL_SIZE
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * size * size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected images of shape (N, 1, 28, 28), got {tuple(x.shape)}")
        return self.head(self.features(x))


@register("cnn")
def build_cnn(
    channels: Sequence[int] = (32, 64),
    hidden_size: int = 128,
    dropout: float = 0.0,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    return ConvClassifier(
        channels=channels,
        hidden_size=hidden_size,
        dropout=dropout,
        num_classes=num_classes,
    )
