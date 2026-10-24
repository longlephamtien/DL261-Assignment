"""Metric contract: hand-computed cases first, then agreement with a reference implementation."""

from __future__ import annotations

import pytest
import torch
from sklearn.metrics import confusion_matrix as sklearn_confusion, jaccard_score

from src.interfaces import IGNORE_INDEX, NUM_CLASSES
from src.metrics import ConfusionMatrix

# A 3x3 two-class example, worked out by hand. One pixel is ignored.
#   targets      predictions
#   0 0 1        0 1 1
#   0 1 1        0 1 0
#   . 1 0        0 1 1          (. is IGNORE_INDEX)
#
# Eight evaluated pixels give the confusion matrix [[2, 2], [1, 3]]:
#   class 0: TP 2, true 4, predicted 3  ->  IoU 2/5,  Dice 2*2/7
#   class 1: TP 3, true 4, predicted 5  ->  IoU 3/6,  Dice 2*3/9
TARGETS = torch.tensor([[0, 0, 1], [0, 1, 1], [IGNORE_INDEX, 1, 0]])
PREDICTIONS = torch.tensor([[0, 1, 1], [0, 1, 0], [0, 1, 1]])


def worked_example() -> ConfusionMatrix:
    matrix = ConfusionMatrix(num_classes=2)
    matrix.update(TARGETS, PREDICTIONS)
    return matrix


def test_confusion_matrix_matches_the_hand_computed_counts() -> None:
    assert worked_example().matrix.tolist() == [[2, 2], [1, 3]]


def test_ignored_pixels_are_excluded_from_every_total() -> None:
    """IGNORE_INDEX marks pixels that carry no evaluated label, so they must not be counted."""
    assert worked_example().matrix.sum().item() == 8, "nine pixels, one of them ignored"


def test_iou_matches_the_hand_computed_values() -> None:
    iou = worked_example().iou()
    assert iou[0].item() == pytest.approx(2 / 5)
    assert iou[1].item() == pytest.approx(3 / 6)
    assert worked_example().mean_iou() == pytest.approx(0.45)


def test_dice_matches_the_hand_computed_values() -> None:
    dice = worked_example().dice()
    assert dice[0].item() == pytest.approx(4 / 7)
    assert dice[1].item() == pytest.approx(6 / 9)
    assert worked_example().mean_dice() == pytest.approx((4 / 7 + 6 / 9) / 2)


def test_dice_and_iou_satisfy_their_algebraic_relation() -> None:
    """Dice = 2 IoU / (1 + IoU) holds per class, so a disagreement means one of them is wrong."""
    matrix = worked_example()
    iou, dice = matrix.iou(), matrix.dice()
    assert torch.allclose(dice, 2 * iou / (1 + iou), equal_nan=True)


def test_pixel_accuracy_matches_the_hand_computed_value() -> None:
    assert worked_example().pixel_accuracy() == pytest.approx(5 / 8)


def test_a_perfect_prediction_scores_one() -> None:
    matrix = ConfusionMatrix(num_classes=NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (2, 8, 8))
    matrix.update(targets, targets.clone())
    assert matrix.mean_iou() == pytest.approx(1.0)
    assert matrix.mean_dice() == pytest.approx(1.0)
    assert matrix.pixel_accuracy() == pytest.approx(1.0)


def test_an_absent_class_is_nan_and_stays_out_of_the_mean() -> None:
    """Scoring a class the split never shows as zero would understate mIoU."""
    matrix = ConfusionMatrix(num_classes=3)
    matrix.update(torch.tensor([0, 1]), torch.tensor([0, 1]))

    iou = matrix.iou()
    assert torch.isnan(iou[2]), "class 2 appears in neither the targets nor the predictions"
    assert matrix.mean_iou() == pytest.approx(1.0), "the two present classes are both perfect"


def test_accumulation_over_batches_equals_one_pass() -> None:
    """A split is scored batch by batch, so the running matrix must not depend on batching."""
    targets = torch.randint(0, NUM_CLASSES, (6, 4, 4))
    predictions = torch.randint(0, NUM_CLASSES, (6, 4, 4))

    whole = ConfusionMatrix()
    whole.update(targets, predictions)

    batched = ConfusionMatrix()
    for start in range(0, 6, 2):
        batched.update(targets[start : start + 2], predictions[start : start + 2])

    assert torch.equal(whole.matrix, batched.matrix)


def test_matrix_agrees_with_scikit_learn() -> None:
    """Reference check: the same counts scikit-learn produces on the evaluated pixels."""
    generator = torch.Generator().manual_seed(0)
    targets = torch.randint(0, NUM_CLASSES, (4, 16, 16), generator=generator)
    predictions = torch.randint(0, NUM_CLASSES, (4, 16, 16), generator=generator)
    targets[targets == 3] = IGNORE_INDEX  # make some pixels ignored

    matrix = ConfusionMatrix()
    matrix.update(targets, predictions)

    keep = (targets != IGNORE_INDEX).flatten()
    reference = sklearn_confusion(
        targets.flatten()[keep].numpy(), predictions.flatten()[keep].numpy(), labels=range(NUM_CLASSES)
    )
    assert matrix.matrix.numpy().tolist() == reference.tolist()


def test_iou_agrees_with_scikit_learn_jaccard() -> None:
    """IoU is the Jaccard index, so scikit-learn's per-class score is the reference."""
    generator = torch.Generator().manual_seed(1)
    targets = torch.randint(0, NUM_CLASSES, (4, 16, 16), generator=generator)
    predictions = torch.randint(0, NUM_CLASSES, (4, 16, 16), generator=generator)
    targets[targets == 5] = IGNORE_INDEX

    matrix = ConfusionMatrix()
    matrix.update(targets, predictions)

    keep = (targets != IGNORE_INDEX).flatten()
    reference = jaccard_score(
        targets.flatten()[keep].numpy(),
        predictions.flatten()[keep].numpy(),
        labels=range(NUM_CLASSES),
        average=None,
        zero_division=0,
    )
    ours = torch.nan_to_num(matrix.iou(), nan=0.0).numpy()
    assert ours == pytest.approx(reference, abs=1e-9)


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="differ"):
        ConfusionMatrix().update(torch.zeros(2, 4, 4, dtype=torch.long), torch.zeros(2, 4, 5, dtype=torch.long))


def test_a_label_outside_the_class_range_is_rejected() -> None:
    """A mask that was never mapped to trainIds would otherwise be counted silently."""
    with pytest.raises(ValueError, match="targets"):
        ConfusionMatrix().update(torch.tensor([NUM_CLASSES]), torch.tensor([0]))


def test_a_prediction_outside_the_class_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="predictions"):
        ConfusionMatrix().update(torch.tensor([0]), torch.tensor([NUM_CLASSES]))


def test_a_fully_ignored_batch_changes_nothing() -> None:
    matrix = ConfusionMatrix()
    matrix.update(torch.full((2, 4, 4), IGNORE_INDEX), torch.zeros(2, 4, 4, dtype=torch.long))
    assert matrix.matrix.sum().item() == 0
    assert matrix.pixel_accuracy() == 0.0
