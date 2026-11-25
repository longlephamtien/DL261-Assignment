"""Modality B: a pretrained text encoder for referring expressions.

The encoder follows :class:`src.interfaces.TextEncoder`: it consumes padded token ids
and an attention mask, and returns one contextual state per input token.  Pooling is
deliberately left to the fusion module because the improved fusion uses the complete
sequence while the simple fusion can pool it into one sentence vector.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# Importing transformers pulls in torch.distributed, which notes on every macOS run that it cannot
# redirect a subprocess's output; nothing here launches one.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

from transformers import AutoModel  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

from ..interfaces import NUM_CLASSES  # noqa: E402
from . import register  # noqa: E402


def load_pretrained(name: str) -> nn.Module:
    """The encoder alone, loaded quietly.

    The checkpoint also holds the masked-language-model head it was pretrained with, which
    AutoModel drops by design; transformers would otherwise report those keys as unexpected on
    every load. The verbosity is restored afterwards, so later warnings still show.
    """
    verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        return AutoModel.from_pretrained(name)
    finally:
        hf_logging.enable_progress_bar()
        hf_logging.set_verbosity(verbosity)


class TextEncoder(nn.Module):
    """Pretrained sentence encoder returning per-token hidden states.

    Representation: the backbone's final ``last_hidden_state`` with shape
    ``(batch, sequence_length, hidden_size)``.  The default pretrained source is
    Hugging Face's ``distilbert-base-uncased`` checkpoint.  ``frozen=True`` keeps
    the backbone in inference mode and excludes it from autograd; ``False`` allows
    the caller to fine-tune it.
    """

    def __init__(self, name: str = "distilbert-base-uncased", frozen: bool = True) -> None:
        super().__init__()
        self.name = name
        self.pretrained_source = f"Hugging Face: {name}"
        self.representation = "last_hidden_state per input token"
        self.frozen = frozen
        self.encoder = load_pretrained(name)
        self.hidden_size = int(self.encoder.config.hidden_size)

        if frozen:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()

    def forward(self, tokens: Tensor, attention: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape (N, L), got {tuple(tokens.shape)}")
        if attention.shape != tokens.shape:
            raise ValueError(
                "attention must have the same shape as tokens; "
                f"got {tuple(attention.shape)} and {tuple(tokens.shape)}"
            )

        # Transformers accepts integer or boolean masks.  Converting here makes the
        # contract explicit and supports the bool mask emitted by the Assignment 3
        # collate function.
        attention_mask = attention.to(dtype=torch.long)
        if self.frozen:
            with torch.no_grad():
                outputs = self.encoder(input_ids=tokens, attention_mask=attention_mask)
        else:
            outputs = self.encoder(input_ids=tokens, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def train(self, mode: bool = True) -> TextEncoder:
        """Keep a frozen backbone in evaluation mode after parent ``train()`` calls."""
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self


def masked_mean(states: Tensor, attention: Tensor) -> Tensor:
    """Average the real tokens of (N, L, D) states into (N, D); padding contributes nothing."""
    weights = attention.unsqueeze(-1).to(states.dtype)
    return (states * weights).sum(1) / weights.sum(1).clamp(min=1)


class TextMaskHead(nn.Module):
    """A sentence vector read out as a coarse two-class mask, then upsampled to the input.

    No pixel is read, so what it can learn is a prior over where and how large the named object
    tends to be: "left", "bottom", "big". The text-only baseline is this head over a fine-tuned
    encoder; the fusion models carry one too, to supervise their text branch on its own.
    """

    def __init__(self, text_width: int, grid: int = 32) -> None:
        super().__init__()
        self.grid = grid
        self.project = nn.Linear(text_width, NUM_CLASSES * grid * grid)

    def forward(self, sentence: Tensor, size: tuple[int, int]) -> Tensor:
        coarse = self.project(sentence).view(-1, NUM_CLASSES, self.grid, self.grid)
        return F.interpolate(coarse, size=size, mode="bilinear", align_corners=False)


class TextOnly(nn.Module):
    """Modality B alone: the pooled expression predicts the mask without seeing the image."""

    def __init__(self, encoder: TextEncoder, grid: int = 32) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = TextMaskHead(encoder.hidden_size, grid)

    def forward(self, images: Tensor, tokens: Tensor, attention: Tensor) -> Tensor:
        # The image is read for its size alone, so the prediction lands at the input resolution.
        sentence = masked_mean(self.encoder(tokens, attention), attention)
        return self.head(sentence, images.shape[-2:])


@register("text-only")
def build_text_only(
    modalities: Sequence[str] = ("text",),
    encoder: str = "distilbert-base-uncased",
    frozen: bool = False,
    grid: int = 32,
) -> nn.Module:
    """Predicts a mask from the expression alone, as a spatial prior. Weak by construction."""
    if list(modalities) != ["text"]:
        raise ValueError(f"the text-only baseline reads the text alone, got {list(modalities)}")
    return TextOnly(TextEncoder(name=encoder, frozen=frozen), grid)
