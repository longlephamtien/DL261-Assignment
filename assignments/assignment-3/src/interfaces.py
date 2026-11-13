"""Shared contract for the Assignment 3 pipeline: text-guided segmentation on RefCOCO."""

from __future__ import annotations

from typing import Literal, TypedDict

from shared.device import count_parameters, describe_hardware, select_device  # noqa: F401  re-exported

Split = Literal["train", "val", "testA", "testB"]
SPLITS: tuple[Split, ...] = ("train", "val", "testA", "testB")
"""The UNC partition, which divides images rather than expressions."""

EVALUATION_SPLITS: tuple[Split, ...] = ("testA", "testB")
"""Read once, after selection on val. testA holds people, testB everything else."""

NUM_CLASSES = 2
"""Foreground and background: an expression names exactly one object."""


class Ref(TypedDict):
    """A referred object, its mask, and the expressions naming it."""

    ref_id: int
    ann_id: int
    image_id: int
    split: str
    category_id: int
    sentences: list[dict]


class Pair(TypedDict):
    """The prediction unit: one expression and the mask it names."""

    ref_id: int
    ann_id: int
    image_id: int
    file_name: str
    sentence: str
    tokens: list[str]
