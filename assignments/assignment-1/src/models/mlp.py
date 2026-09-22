from __future__ import annotations

from typing import Sequence

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
    def __init__(
        self,
        in_features: int = 784,
        hidden_sizes: Sequence[int] = (512, 256),
        num_classes: int = 10,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        act_cls = ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"unsupported activation: {activation}")

        layers: list[nn.Module] = []
        prev_dim = in_features
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        return self.net(x)


@register("mlp")
def build_mlp(
    in_features: int = 784,
    hidden_sizes: Sequence[int] = (512, 256),
    num_classes: int = 10,
    activation: str = "relu",
    dropout: float = 0.0,
    **kwargs,
) -> nn.Module:
    return MLPClassifier(
        in_features=in_features,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        activation=activation,
        dropout=dropout,
    )
