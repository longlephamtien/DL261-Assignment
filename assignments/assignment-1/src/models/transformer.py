"""Transformer encoder over image tokens: rows, columns, or patches."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data import to_sequence
from ..interfaces import IMAGE_SHAPE, NUM_CLASSES, Representation
from . import register

SEQUENCE_REPRESENTATIONS = ("rows", "columns", "patches")
POOLINGS = ("cls", "mean")


def token_shape(representation: Representation, patch_size: int) -> tuple[int, int]:
    """Tokens per image and features per token, for one image of `interfaces.IMAGE_SHAPE`."""
    _, height, width = IMAGE_SHAPE
    if representation == "rows":
        return height, width
    if representation == "columns":
        return width, height
    if representation == "patches":
        if height % patch_size or width % patch_size:
            raise ValueError(f"{height}x{width} images are not divisible by patch_size {patch_size}")
        return (height // patch_size) * (width // patch_size), patch_size * patch_size
    raise ValueError(
        f"'{representation}' is not a sequence; expected one of {', '.join(SEQUENCE_REPRESENTATIONS)}"
    )


class TransformerClassifier(nn.Module):
    """Tokens, projection, positional encoding, self-attention, then a readout.

    For a batch of N images split into T tokens of F features, with model width D:

        to_sequence     (N, 1, 28, 28) -> (N, T, F)
        projection      (N, T, F)      -> (N, T, D)
        class token     (N, T, D)      -> (N, T + 1, D), when pooling is "cls"
        positions       added in place, one learned vector per position
        attention       query, key, and value are the same (N, T + 1, D) tensor,
                        each head attends over the token axis and returns (N, T + 1, D)
        readout         (N, T + 1, D)  -> (N, D) -> (N, 10)
    """

    def __init__(
        self,
        representation: Representation = "patches",
        patch_size: int = 4,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        pooling: str = "cls",
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"unsupported pooling '{pooling}'; expected one of {', '.join(POOLINGS)}")
        if d_model % num_heads:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")

        tokens, features = token_shape(representation, patch_size)
        self.representation = representation
        self.patch_size = patch_size
        self.pooling = pooling

        self.projection = nn.Linear(features, d_model)
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pooling == "cls" else None
        self.positions = nn.Parameter(torch.zeros(1, tokens + (pooling == "cls"), d_model))
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.positions, std=0.02)
        if self.class_token is not None:
            nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected images of shape (N, 1, 28, 28), got {tuple(x.shape)}")

        hidden = self.projection(to_sequence(x, self.representation, self.patch_size))
        if self.class_token is not None:
            hidden = torch.cat([self.class_token.expand(len(hidden), -1, -1), hidden], dim=1)
        hidden = self.encoder(self.dropout(hidden + self.positions))
        pooled = hidden[:, 0] if self.class_token is not None else hidden.mean(dim=1)
        return self.head(self.norm(pooled))


@register("transformer")
def build_transformer(
    representation: Representation = "patches",
    patch_size: int = 4,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    ff_dim: int = 256,
    dropout: float = 0.1,
    pooling: str = "cls",
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    return TransformerClassifier(
        representation=representation,
        patch_size=patch_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        pooling=pooling,
        num_classes=num_classes,
    )
