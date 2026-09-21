"""Dataset, splits, loaders, and sequence adapters.

    python -m src.data --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from shared import io as shared_io

from .interfaces import CLASS_NAMES, Representation, SplitFile, select_device
from .utils import load_config, resolve_path


def prepare_dataset(config: dict) -> Path:
    """Download Fashion-MNIST into `dataset.root` and return that path."""
    dataset = config["dataset"]
    root = resolve_path(dataset["root"])
    for train in (True, False):
        torchvision.datasets.FashionMNIST(root=str(root), train=train, download=dataset["download"])
    return root


def build_split(data_dir: Path, seed: int, val_size: int, stratified: bool = True) -> SplitFile:
    """Split the official training set into stratified train and validation indices."""
    labels = torchvision.datasets.FashionMNIST(root=str(data_dir), train=True).targets.numpy()
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
    """Channel mean and standard deviation computed on the training split only."""
    images = torchvision.datasets.FashionMNIST(root=str(data_dir), train=True).data[split["train"]]
    images = images.float() / 255.0
    return float(images.mean()), float(images.std())


def build_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Loaders for 'train', 'val', and 'test'; stores train-split stats in `dataset.normalization`."""
    dataset = config["dataset"]
    loader = config["loader"]
    data_dir = resolve_path(dataset["root"])
    split = load_split(resolve_path(dataset["split_file"]))

    mean, std = normalization_stats(data_dir, split)
    dataset["normalization"] = {"mean": mean, "std": std}

    normalize = [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    augment = [
        transforms.ToTensor(),
        transforms.RandomCrop(28, padding=2, padding_mode="edge"),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((mean,), (std,)),
    ]
    train_transform = transforms.Compose(augment if loader["augment"] else normalize)
    eval_transform = transforms.Compose(normalize)

    def official(train: bool, transform: transforms.Compose) -> Dataset:
        return torchvision.datasets.FashionMNIST(root=str(data_dir), train=train, transform=transform)

    def make_loader(data: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            data,
            batch_size=loader["batch_size"],
            shuffle=shuffle,
            num_workers=loader["num_workers"],
            pin_memory=select_device().type == "cuda",
        )

    return {
        "train": make_loader(Subset(official(True, train_transform), split["train"]), shuffle=True),
        "val": make_loader(Subset(official(True, eval_transform), split["val"]), shuffle=False),
        "test": make_loader(official(False, eval_transform), shuffle=False),
    }


def export_example_batch(loaders: dict[str, DataLoader], path: Path) -> None:
    """Write shapes, dtypes, and value ranges of one preprocessed validation batch."""
    images, labels = next(iter(loaders["val"]))

    def describe(tensor: Tensor) -> dict:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
        }

    shared_io.write_json(path, {"images": describe(images), "labels": describe(labels)})


def to_sequence(images: Tensor, representation: Representation, patch_size: int = 4) -> Tensor:
    """Images (N, 1, 28, 28) to (N, 28, 28) for rows and columns, or (N, T, patch_size ** 2) for patches."""
    n, _, height, width = images.shape
    if representation == "image":
        return images
    if representation == "rows":
        return images.squeeze(1)
    if representation == "columns":
        return images.squeeze(1).transpose(1, 2).contiguous()
    if representation == "patches":
        if height % patch_size or width % patch_size:
            raise ValueError(f"{height}x{width} images are not divisible by patch_size {patch_size}")
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        return patches.contiguous().view(n, -1, patch_size * patch_size)
    raise ValueError(f"unknown representation '{representation}'")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Fashion-MNIST, write the split, export an example batch.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args(argv)

    config = load_config(resolve_path(args.config))
    dataset = config["dataset"]
    root = prepare_dataset(config)
    split = build_split(root, config["seed"], dataset["val_size"], dataset["stratified"])
    write_split(split, resolve_path(dataset["split_file"]))
    export_example_batch(build_dataloaders(config), resolve_path("results/eda/example_batch.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
