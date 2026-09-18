"""Dataset, splits, loaders, and sequence adapters.

    python -m src.data --config configs/base.yaml

Split is implemented (#13). Download, DataLoaders, and sequence adapters: #14.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader

from shared import io as shared_io

from .interfaces import CLASS_NAMES, Representation, SplitFile


def prepare_dataset(config: dict) -> Path:
    """Download Fashion-MNIST into `dataset.root` and return that path. Issue #14.

    The download directory is gitignored; the dataset is public and re-downloadable, so it
    is never committed or mirrored.
    """
    raise NotImplementedError("issue #14")


def build_split(data_dir: Path, seed: int, val_size: int, stratified: bool = True) -> SplitFile:
    """Partition the official 60,000-image training set into train and validation indices.

    `data_dir` must already hold the downloaded training set; see `prepare_dataset`.
    """
    labels = np.array(torchvision.datasets.FashionMNIST(root=str(data_dir), train=True).targets)
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=labels if stratified else None,
    )

    assert set(train_idx) & set(val_idx) == set()
    assert set(train_idx) | set(val_idx) == set(indices.tolist())

    return SplitFile(
        dataset="fashion-mnist",
        seed=seed,
        stratified=stratified,
        train=sorted(train_idx.tolist()),
        val=sorted(val_idx.tolist()),
        counts={"train": _class_counts(labels[train_idx]), "val": _class_counts(labels[val_idx])},
    )


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {CLASS_NAMES[label]: int(count) for label, count in zip(*np.unique(labels, return_counts=True))}


def write_split(split: SplitFile, path: Path) -> None:
    """Write a split file to `splits/`; committed so every model shares one partition."""
    shared_io.write_json(path, split)


def load_split(path: Path) -> SplitFile:
    """Read a committed split file."""
    return shared_io.read_json(path)


def normalization_stats(data_dir: Path, split: SplitFile) -> tuple[float, float]:
    """Channel mean and standard deviation computed on the training split only. Issue #14."""
    raise NotImplementedError("issue #14")


def build_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Loaders keyed by 'train', 'val', and 'test', yielding batches per `interfaces.Batch`.

    Augmentation applies to the training loader only and is identical for every model.
    `pin_memory` should follow `interfaces.select_device().type == "cuda"`, since it only
    speeds up CUDA transfers and MPS does not support it. `num_workers` > 0 needs the
    `if __name__ == "__main__":` guard on Windows, already present in every CLI entry point
    here. Issue #14.
    """
    raise NotImplementedError("issue #14")


def to_sequence(images: Tensor, representation: Representation, patch_size: int = 4) -> Tensor:
    """Turn images (N, 1, 28, 28) into a sequence (N, T, F).

    rows     -> (N, 28, 28)
    columns  -> (N, 28, 28)
    patches  -> (N, (28 / patch_size) ** 2, patch_size ** 2)

    Issue #14.
    """
    raise NotImplementedError("issue #14")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Fashion-MNIST and write the shared split.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args(argv)

    from .utils import load_config

    config = load_config(args.config)
    dataset = config["dataset"]
    root = prepare_dataset(config)
    split = build_split(root, config["seed"], dataset["val_size"], dataset["stratified"])
    write_split(split, Path(dataset["split_file"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
