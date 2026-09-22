"""Multilayer perceptron: the same flattened input as the linear model, plus hidden layers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from . import register

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


class MLPClassifier(nn.Module):
    """Depth, width, activation, and dropout all come from configuration."""

    def __init__(
        self,
        in_features: int = 784,
        hidden_sizes: Sequence[int] = (512, 256),
        num_classes: int = 10,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("an MLP needs at least one hidden layer")
        if activation.lower() not in ACTIVATIONS:
            raise ValueError(f"unsupported activation '{activation}'; expected one of {', '.join(ACTIVATIONS)}")

        build_activation = ACTIVATIONS[activation.lower()]
        layers: list[nn.Module] = []
        width = in_features
        for hidden in hidden_sizes:
            layers += [nn.Linear(width, hidden), build_activation()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            width = hidden
        layers.append(nn.Linear(width, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(torch.flatten(x, 1))


@register("mlp")
def build_mlp(
    in_features: int = 784,
    hidden_sizes: Sequence[int] = (512, 256),
    num_classes: int = 10,
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Module:
    return MLPClassifier(
        in_features=in_features,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        activation=activation,
        dropout=dropout,
    )
