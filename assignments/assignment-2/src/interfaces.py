"""Shared contract for the Assignment 2 pipeline: semantic segmentation on Cityscapes."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from torch import Tensor

from shared.device import count_parameters, describe_hardware, select_device  # noqa: F401  re-exported

IGNORE_INDEX = 255
"""Label of pixels excluded from loss and metrics, as Cityscapes writes unlabelled regions."""

CLASS_NAMES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
"""The 19 Cityscapes evaluation classes, in trainId order."""

NUM_CLASSES = len(CLASS_NAMES)

Split = Literal["train", "val", "test"]
EVALUATION_SPLITS: tuple[Split, ...] = ("val", "test")
"""Splits a finished run is scored on. Cityscapes withholds test labels, so val is the reported split."""

Batch = tuple[Tensor, Tensor]
"""Images of shape (N, 3, H, W) as float32, and masks of shape (N, H, W) as int64.

Images are normalized with statistics computed on the training split only. Mask values are
trainIds in [0, 19), or IGNORE_INDEX for pixels that carry no evaluated label.
"""


class SegmentationModule(Protocol):
    """A model returns per-pixel logits of shape (N, 19, H, W) at the input resolution."""

    def forward(self, images: Tensor) -> Tensor: ...


class SplitFile(TypedDict):
    """Contents of `splits/<dataset>-seed<seed>.json`, committed to the repository.

    The split unit is the city, so no street scene can appear in two partitions.
    """

    dataset: str
    seed: int
    split_unit: Literal["city"]
    train: list[str]
    val: list[str]
    cities: dict[str, list[str]]
    counts: dict[str, int]


class EpochRecord(TypedDict):
    """One entry of `history.json`, appended after every epoch."""

    epoch: int
    train_loss: float
    train_miou: float
    val_loss: float
    val_miou: float
    val_dice: float
    seconds: float
    learning_rate: float


class RunSummary(TypedDict):
    """Contents of `summary.json`, written once per run."""

    run_id: str
    model: str
    backbone: str
    seed: int
    parameters: int
    epochs_trained: int
    best_epoch: int
    checkpoint_metric: str
    checkpoint_value: float
    training_seconds: float
    commit: str
    hardware: str


class EvaluationReport(TypedDict):
    """Contents of `results/evaluation/<run_id>/<split>/report.json`."""

    run_id: str
    model: str
    backbone: str
    split: str
    miou: float
    dice: float
    pixel_accuracy: float
    per_class_iou: dict[str, float]
    per_class_dice: dict[str, float]
    confusion_matrix: list[list[int]]
    parameters: int
    training_seconds: float
    inference_ms_per_image: dict[str, float]
