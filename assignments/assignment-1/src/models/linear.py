"""Linear softmax classifier: the capacity floor for the comparison."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import register


class LinearClassifier(nn.Module):
    """Flattened pixels mapped to raw logits by one affine layer."""

    def __init__(self, in_features: int = 784, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(torch.flatten(x, 1))


@register("linear")
def build_linear(in_features: int = 784, num_classes: int = 10) -> nn.Module:
    return LinearClassifier(in_features=in_features, num_classes=num_classes)
