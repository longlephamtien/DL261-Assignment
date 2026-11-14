"""The training pass, checked against a stand-in model so it does not wait on the encoders."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.interfaces import NUM_CLASSES
from src.train import CombinedLoss, DiceLoss, build_criterion, build_optimizer, run_epoch


class Stub(nn.Module):
    """Takes the three arguments every model takes and reads only the image."""

    def __init__(self, size: int = 8) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, NUM_CLASSES, 1)

    def forward(self, images, tokens, attention):
        return self.head(images)


def batches(count: int = 2, size: int = 8, length: int = 3) -> DataLoader:
    samples = [{
        "images": torch.randn(2, 3, size, size),
        "tokens": torch.randint(1, 10, (2, length)),
        "attention": torch.ones(2, length, dtype=torch.bool),
        "masks": torch.randint(0, NUM_CLASSES, (2, size, size)),
        "ref_ids": torch.tensor([0, 1]),
    } for _ in range(count)]
    return DataLoader(samples, batch_size=None, collate_fn=lambda x: x)


def test_a_validation_pass_reports_loss_and_both_metrics() -> None:
    result = run_epoch(Stub(), batches(), CombinedLoss(), torch.device("cpu"))
    assert set(result) == {"loss", "miou", "dice"}
    assert all(isinstance(v, float) for v in result.values())


def test_a_validation_pass_leaves_the_weights_alone() -> None:
    model = Stub()
    before = model.head.weight.detach().clone()
    run_epoch(model, batches(), CombinedLoss(), torch.device("cpu"))
    assert torch.equal(before, model.head.weight)


def test_passing_an_optimizer_updates_the_weights() -> None:
    model = Stub()
    before = model.head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    run_epoch(model, batches(), CombinedLoss(), torch.device("cpu"), optimizer)
    assert not torch.equal(before, model.head.weight)


def test_the_pass_accepts_any_model_taking_the_three_arguments() -> None:
    """A unimodal baseline ignores an argument; the pass must not care which."""

    class TextOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(1, NUM_CLASSES * 8 * 8)

        def forward(self, images, tokens, attention):
            pooled = attention.float().sum(1, keepdim=True)
            return self.head(pooled).view(-1, NUM_CLASSES, 8, 8)

    result = run_epoch(TextOnly(), batches(), CombinedLoss(), torch.device("cpu"))
    assert result["loss"] > 0


@pytest.mark.parametrize("name,kind", [("cross_entropy", nn.CrossEntropyLoss),
                                       ("dice", DiceLoss), ("combined", CombinedLoss)])
def test_every_configured_loss_builds(name: str, kind: type) -> None:
    assert isinstance(build_criterion({"train": {"loss": name}}), kind)


def test_an_unknown_loss_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported loss"):
        build_criterion({"train": {"loss": "focal"}})


def test_dice_is_zero_on_a_perfect_prediction() -> None:
    logits = torch.stack([torch.full((1, 4, 4), -10.0), torch.full((1, 4, 4), 10.0)], dim=1)
    targets = torch.ones(1, 4, 4, dtype=torch.long)
    assert DiceLoss()(logits, targets).item() == pytest.approx(0.0, abs=1e-3)


def test_the_optimizer_skips_frozen_parameters() -> None:
    """A frozen text encoder must not appear in the optimizer's parameter groups."""
    model = Stub()
    model.head.weight.requires_grad_(False)
    optimizer = build_optimizer({"train": {"optimizer": "adamw"}}, model)
    assert all(p is not model.head.weight for group in optimizer.param_groups for p in group["params"])
