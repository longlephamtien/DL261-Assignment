"""The modality-B encoder: per-token states, frozen and fine-tuned modes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models import text


class FakeTextBackbone(nn.Module):
    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_text_encoder_returns_per_token_embeddings_for_synthetic_batch(monkeypatch) -> None:
    monkeypatch.setattr(text.AutoModel, "from_pretrained", lambda name: FakeTextBackbone())

    encoder = text.TextEncoder(name="synthetic-checkpoint", frozen=True)
    tokens = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long)
    attention = tokens.ne(0)

    embeddings = encoder(tokens, attention)

    assert embeddings.shape == (2, 3, 12)
    assert embeddings.dtype == torch.float32
    assert not embeddings.requires_grad
    assert all(not parameter.requires_grad for parameter in encoder.encoder.parameters())


def test_text_encoder_can_be_fine_tuned(monkeypatch) -> None:
    monkeypatch.setattr(text.AutoModel, "from_pretrained", lambda name: FakeTextBackbone())

    encoder = text.TextEncoder(name="synthetic-checkpoint", frozen=False)
    output = encoder(torch.tensor([[1, 2]]), torch.tensor([[True, True]]))

    assert output.shape == (1, 2, 12)
    assert output.requires_grad
    assert any(parameter.requires_grad for parameter in encoder.encoder.parameters())


def test_the_text_only_baseline_predicts_a_mask_without_reading_the_image(monkeypatch) -> None:
    monkeypatch.setattr(text.AutoModel, "from_pretrained", lambda name: FakeTextBackbone())
    model = text.build_text_only(encoder="synthetic-checkpoint")
    tokens = torch.tensor([[1, 2, 0]])
    attention = tokens.ne(0)
    first = model(torch.zeros(1, 3, 64, 48), tokens, attention)
    second = model(torch.randn(1, 3, 64, 48), tokens, attention)
    assert first.shape == (1, 2, 64, 48)
    assert torch.equal(first, second)


def test_the_text_only_baseline_fine_tunes_its_encoder(monkeypatch) -> None:
    monkeypatch.setattr(text.AutoModel, "from_pretrained", lambda name: FakeTextBackbone())
    model = text.build_text_only(encoder="synthetic-checkpoint", frozen=False)
    model(torch.zeros(1, 3, 8, 8), torch.tensor([[1, 2]]), torch.tensor([[True, True]])).sum().backward()
    assert model.encoder.encoder.embedding.weight.grad is not None


def test_the_text_only_builder_refuses_a_second_modality() -> None:
    with pytest.raises(ValueError, match="text alone"):
        text.build_text_only(modalities=["image", "text"])
