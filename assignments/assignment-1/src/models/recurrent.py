from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data import to_sequence
from ..interfaces import NUM_CLASSES, Representation
from . import register


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        cell_type: str = "gru",
        representation: Representation = "rows",
        patch_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.cell_type = cell_type.lower()
        self.representation = representation
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if representation in ("rows", "columns"):
            self.input_size = 28
            self.sequence_length = 28
        elif representation == "patches":
            if 28 % patch_size != 0:
                raise ValueError(f"28x28 image not divisible by patch_size {patch_size}")
            self.input_size = patch_size * patch_size
            self.sequence_length = (28 // patch_size) ** 2
        else:
            raise ValueError(f"Unsupported representation: '{representation}'")

        rnn_cls = nn.LSTM if self.cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        num_directions = 2 if bidirectional else 1
        classifier_input_dim = hidden_size * num_directions
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        x = to_sequence(images, self.representation, patch_size=self.patch_size)

        if self.cell_type == "lstm":
            out, (h_n, c_n) = self.rnn(x)
        else:
            out, h_n = self.rnn(x)

        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            rep = torch.cat([h_forward, h_backward], dim=1)
        else:
            rep = h_n[-1, :, :]

        return self.classifier(rep)


@register("recurrent")
def build_recurrent(**kwargs) -> nn.Module:
    return RecurrentClassifier(**kwargs)