"""Shared contract for the Assignment 1 pipeline."""

from __future__ import annotations

import platform
import subprocess
from typing import Literal, Protocol, TypedDict

import psutil
import torch
from torch import Tensor, nn

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
    """Contents of `results/evaluation/<run_id>.json`, one per model in the comparison."""

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


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters, reported for every model in the comparison."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_device() -> torch.device:
    """CUDA if available, else Apple Silicon MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_hardware() -> str:
    """One-line device description recorded in every run summary.

    Training and inference times are only comparable across models when they come from the
    same machine, so the string names the accelerator, the host processor, and its memory.
    """
    device = select_device()
    host = f"{_processor()}, {psutil.cpu_count(logical=False)} cores, {_gigabytes(psutil.virtual_memory().total)} RAM"
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        return f"cuda: {properties.name}, {_gigabytes(properties.total_memory)} VRAM; host {host}"
    if device.type == "mps":
        return f"mps: {host}"
    return f"cpu: {host}"


def _gigabytes(size: int) -> str:
    return f"{size / 1024 ** 3:.0f} GB"


def _processor() -> str:
    """Chip name, which `platform.processor()` reports only as the architecture on macOS."""
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or platform.machine()
