"""Modality B: a pretrained text encoder for referring expressions.

The encoder follows :class:`src.interfaces.TextEncoder`: it consumes padded token ids
and an attention mask, and returns one contextual state per input token.  Pooling is
deliberately left to the fusion module because the improved fusion uses the complete
sequence while the simple fusion can pool it into one sentence vector.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import AutoModel

from . import register


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
        self.encoder = AutoModel.from_pretrained(name)
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


@register("text-only")
def build_text_only(**kwargs) -> nn.Module:
    """Predicts a mask from the expression alone, as a spatial prior. Weak by construction."""
    raise NotImplementedError("#101 trains the modality-B baseline")
