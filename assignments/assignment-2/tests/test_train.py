"""Training contract: the losses, and what one pass must and must not change."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.interfaces import IGNORE_INDEX, NUM_CLASSES
from src.train import DiceLoss, build_criterion, run_epoch


class Tiny(nn.Module):
    """A 1x1 convolution: enough to have gradients, cheap enough for a unit test."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, NUM_CLASSES, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(images)


def loader(batch: int = 2, size: int = 8) -> DataLoader:
    images = torch.randn(4, 3, size, size)
    targets = torch.randint(0, NUM_CLASSES, (4, size, size))
    return DataLoader(TensorDataset(images, targets), batch_size=batch)


def test_cross_entropy_is_selected_by_default() -> None:
    criterion = build_criterion({"train": {}})
    assert isinstance(criterion, nn.CrossEntropyLoss)


def test_cross_entropy_ignores_the_unlabelled_pixels() -> None:
    """Without ignore_index the loss would index class 255 and the run would be meaningless."""
    assert build_criterion({"train": {"loss": "cross_entropy"}}).ignore_index == IGNORE_INDEX


def test_dice_is_selected_by_name() -> None:
    assert isinstance(build_criterion({"train": {"loss": "dice"}}), DiceLoss)


def test_an_unknown_loss_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported loss"):
        build_criterion({"train": {"loss": "focal"}})


def test_cross_entropy_runs_on_a_mask_full_of_ignored_pixels() -> None:
    logits = torch.randn(1, NUM_CLASSES, 4, 4)
    targets = torch.full((1, 4, 4), IGNORE_INDEX)
    assert torch.isnan(build_criterion({"train": {}})(logits, targets))


def test_dice_returns_zero_loss_for_a_perfect_prediction() -> None:
    targets = torch.randint(0, NUM_CLASSES, (2, 4, 4))
    logits = nn.functional.one_hot(targets, NUM_CLASSES).permute(0, 3, 1, 2).float() * 50
    assert DiceLoss()(logits, targets).item() == pytest.approx(0.0, abs=1e-3)


def test_dice_scores_a_wrong_prediction_worse_than_a_right_one() -> None:
    targets = torch.zeros(2, 4, 4, dtype=torch.long)
    right = nn.functional.one_hot(targets, NUM_CLASSES).permute(0, 3, 1, 2).float() * 50
    wrong = nn.functional.one_hot(torch.ones_like(targets), NUM_CLASSES).permute(0, 3, 1, 2).float() * 50
    assert DiceLoss()(wrong, targets) > DiceLoss()(right, targets)


def test_dice_ignores_the_unlabelled_pixels() -> None:
    """Pixels marked IGNORE_INDEX must not enter the per-class sums."""
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    logits = nn.functional.one_hot(targets, NUM_CLASSES).permute(0, 3, 1, 2).float() * 50

    ignored = targets.clone()
    ignored[:, 2:] = IGNORE_INDEX
    assert DiceLoss()(logits, ignored).item() == pytest.approx(DiceLoss()(logits, targets).item(), abs=1e-3)


def test_dice_survives_a_fully_ignored_batch() -> None:
    logits = torch.randn(1, NUM_CLASSES, 4, 4, requires_grad=True)
    loss = DiceLoss()(logits, torch.full((1, 4, 4), IGNORE_INDEX))

    assert loss.item() == 0.0
    loss.backward()  # must stay connected to the graph, or the step would crash


def test_a_pass_without_an_optimizer_leaves_the_weights_alone() -> None:
    """Validation must not train; a leaked optimizer would invalidate every reported score."""
    model = Tiny()
    before = model.head.weight.clone()

    run_epoch(model, loader(), build_criterion({"train": {}}), torch.device("cpu"))

    assert torch.equal(before, model.head.weight)


def test_a_pass_with_an_optimizer_updates_the_weights() -> None:
    model = Tiny()
    before = model.head.weight.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    run_epoch(model, loader(), build_criterion({"train": {}}), torch.device("cpu"), optimizer)

    assert not torch.equal(before, model.head.weight)


def test_a_pass_reports_loss_miou_and_dice() -> None:
    loss, miou, dice = run_epoch(Tiny(), loader(), build_criterion({"train": {}}), torch.device("cpu"))

    assert loss > 0
    assert 0.0 <= miou <= 1.0
    assert 0.0 <= dice <= 1.0


def test_a_pass_leaves_the_model_in_eval_mode_when_not_training() -> None:
    model = Tiny()
    run_epoch(model, loader(), build_criterion({"train": {}}), torch.device("cpu"))
    assert not model.training
