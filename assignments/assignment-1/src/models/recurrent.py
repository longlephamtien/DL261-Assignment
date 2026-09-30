from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data import to_sequence, token_shape
from ..interfaces import NUM_CLASSES, Representation
from . import register

CELLS: dict[str, type[nn.RNNBase]] = {"gru": nn.GRU, "lstm": nn.LSTM}


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
        if cell_type.lower() not in CELLS:
            raise ValueError(f"unsupported cell_type '{cell_type}'; expected one of {', '.join(CELLS)}")

        self.cell_type = cell_type.lower()
        self.representation = representation
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.sequence_length, self.input_size = token_shape(representation, patch_size)

        self.rnn = CELLS[self.cell_type](
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