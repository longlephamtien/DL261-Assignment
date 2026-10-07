"""Shared contract for the Assignment 1 pipeline."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from torch import Tensor

from shared.device import count_parameters, describe_hardware, select_device  # noqa: F401  re-exported

NUM_CLASSES = 10
IMAGE_SHAPE = (1, 28, 28)
CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

Split = Literal["train", "val", "test"]
Representation = Literal["image", "rows", "columns", "patches"]

EVALUATION_SPLITS: tuple[Split, ...] = ("val", "test")
"""Splits a finished run is scored on. Training data is never reported as a result."""

Batch = tuple[Tensor, Tensor]
"""Images of shape (N, 1, 28, 28) as float32, and labels of shape (N,) as int64.

Images are normalized with statistics computed on the training split only. A model that
consumes sequences receives the same batch and calls `data.to_sequence` itself.
"""


class ClassifierModule(Protocol):
    """A model returns raw logits of shape (N, 10); softmax belongs to the loss."""

    def forward(self, images: Tensor) -> Tensor: ...


class SplitFile(TypedDict):
    """Contents of `splits/<dataset>-seed<seed>.json`, committed to the repository."""

    dataset: str
    seed: int
    stratified: bool
    train: list[int]
    val: list[int]
    counts: dict[str, dict[str, int]]


class EpochRecord(TypedDict):
    """One entry of `history.json`, appended after every epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    train_macro_f1: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    seconds: float
    learning_rate: float


class RunSummary(TypedDict):
    """Contents of `summary.json`, written once per run."""

    run_id: str
    model: str
    representation: Representation
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
    representation: Representation
    split: Split
    accuracy: float
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    parameters: int
    training_seconds: float
    inference_ms_per_image: dict[str, float]
