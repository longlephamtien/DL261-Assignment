"""Shared contract for the Assignment 3 pipeline: text-guided segmentation on RefCOCO.

Write against this module, not against another module's implementation, so the data, encoder,
fusion and evaluation work can proceed in parallel.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from torch import Tensor

from shared.device import count_parameters, describe_hardware, select_device  # noqa: F401  re-exported

Split = Literal["train", "val", "testA", "testB"]
SPLITS: tuple[Split, ...] = ("train", "val", "testA", "testB")
"""The UNC partition, which divides images rather than expressions."""

EVALUATION_SPLITS: tuple[Split, ...] = ("testA", "testB")
"""Read once, after selection on val. testA holds people, testB everything else."""

NUM_CLASSES = 2
"""Foreground and background: an expression names exactly one object."""

PAD_TOKEN = 0
"""Token id that carries no word. Attention and the text pooler must ignore it."""


class Ref(TypedDict):
    """A referred object as `refs(unc).p` stores it, with the expressions naming it."""

    ref_id: int
    ann_id: int
    image_id: int
    split: str
    category_id: int
    sentences: list[dict]


class Pair(TypedDict):
    """The prediction unit: one expression and the mask of the object it names."""

    ref_id: int
    ann_id: int
    image_id: int
    file_name: str
    sentence: str
    tokens: list[str]


class Batch(TypedDict):
    """What the loader yields. The two modalities are aligned row by row.

    `images` is (N, 3, H, W) float32, normalized. `tokens` is (N, L) int64 padded with
    PAD_TOKEN, and `attention` is (N, L) bool marking the real tokens. `masks` is (N, H, W)
    int64 holding 1 on the referred object and 0 elsewhere.
    """

    images: Tensor
    tokens: Tensor
    attention: Tensor
    masks: Tensor
    ref_ids: Tensor


class VisionEncoder(Protocol):
    """Modality A. Returns a feature map (N, C, H', W') keeping spatial layout for the decoder."""

    def forward(self, images: Tensor) -> Tensor: ...


class TextEncoder(Protocol):
    """Modality B. Returns per-token states (N, L, D); pooling is the fusion module's choice."""

    def forward(self, tokens: Tensor, attention: Tensor) -> Tensor: ...


class FusionModule(Protocol):
    """Combines the two streams into one feature map the mask head can read.

    Takes the vision feature map, the text states, and the token mask, and returns a map of the
    same spatial size as the vision input.
    """

    def forward(self, vision: Tensor, text: Tensor, attention: Tensor) -> Tensor: ...


class SegmentationModule(Protocol):
    """The whole model: per-pixel logits (N, 2, H, W) at the input resolution.

    A unimodal baseline implements this too, ignoring whichever argument it does not read, so
    all four models of section 29.1 share one training and evaluation path.
    """

    def forward(self, images: Tensor, tokens: Tensor, attention: Tensor) -> Tensor: ...


class SplitFile(TypedDict):
    """Contents of `splits/refcoco-unc.json`, committed to the repository.

    The split unit is the image, so every expression naming an object in one image stays in one
    partition.
    """

    dataset: str
    partition: str
    split_unit: Literal["image"]
    images: dict[str, list[int]]
    counts: dict[str, dict[str, int]]


class EpochRecord(TypedDict):
    """One entry of `history.json`, appended after every epoch."""

    epoch: int
    train_loss: float
    train_miou: float
    train_dice: float
    val_loss: float
    val_miou: float
    val_dice: float
    seconds: float
    learning_rate: float


class RunSummary(TypedDict):
    """Contents of `summary.json`, written once per run."""

    run_id: str
    model: str
    modalities: list[str]
    fusion: str
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
    modalities: list[str]
    fusion: str
    split: str
    miou: float
    dice: float
    precision_at_50: float
    per_category_iou: dict[str, float]
    parameters: int
    training_seconds: float
    inference_ms_per_pair: dict[str, float]
    peak_memory_bytes: int
