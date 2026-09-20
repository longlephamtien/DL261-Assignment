"""Dataset, splits, loaders, and sequence adapters.

    python -m src.data --config configs/base.yaml

Split is implemented (#13). Download, DataLoaders, and sequence adapters: #14.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from shared import io as shared_io

from .interfaces import (
    CLASS_NAMES,
    Representation,
    SplitFile,
    select_device,
)


def prepare_dataset(config: dict) -> Path:
    """Download Fashion-MNIST into `dataset.root` and return that path. Issue #14.

    The download directory is gitignored; the dataset is public and re-downloadable, so it
    is never committed or mirrored.
    """
    dataset_cfg = config.get("dataset", {})
    root = Path(dataset_cfg.get("root", "data"))
    download = dataset_cfg.get("download", True)

    # Tải cả train và test set
    torchvision.datasets.FashionMNIST(root=str(root), train=True, download=download)
    torchvision.datasets.FashionMNIST(root=str(root), train=False, download=download)
    return root


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
    raw_train = torchvision.datasets.FashionMNIST(
        root=str(data_dir), train=True, download=False
    )
    # raw_train.data có shape (60000, 28, 28) dạng uint8 [0, 255]
    train_images = raw_train.data[split["train"]].float() / 255.0
    mean = float(train_images.mean().item())
    std = float(train_images.std().item())
    return mean, std


def build_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Loaders keyed by 'train', 'val', and 'test', yielding batches per `interfaces.Batch`.

    Augmentation applies to the training loader only and is identical for every model.
    `pin_memory` should follow `interfaces.select_device().type == "cuda"`, since it only
    speeds up CUDA transfers and MPS does not support it. `num_workers` > 0 needs the
    `if __name__ == "__main__":` guard on Windows, already present in every CLI entry point
    here. Issue #14.
    """
    dataset_cfg = config.get("dataset", {})
    loader_cfg = config.get("loader", {})

    data_dir = prepare_dataset(config)

    # Tìm đường dẫn file split an toàn
    split_file_cfg = Path(dataset_cfg.get("split_file", "splits/fashion-mnist-seed0.json"))
    if split_file_cfg.is_file():
        split_path = split_file_cfg
    elif (Path("assignments/assignment-1") / split_file_cfg).is_file():
        split_path = Path("assignments/assignment-1") / split_file_cfg
    else:
        split_path = split_file_cfg

    split = load_split(split_path)

    # 1. Tính mean, std chỉ trên tập train
    mean, std = normalization_stats(data_dir, split)

    # 2. Xây dựng transforms
    base_eval_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ]

    if loader_cfg.get("augment", False):
        # ToTensor trước để đưa về dạng Tensor [1, 28, 28], sau đó crop và flip an toàn
        train_transforms = [
            transforms.ToTensor(),
            transforms.RandomCrop(28, padding=2, padding_mode="edge"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize((mean,), (std,)),
        ]
    else:
        train_transforms = base_eval_transforms

    train_tf = transforms.Compose(train_transforms)
    eval_tf = transforms.Compose(base_eval_transforms)

    # 3. Tạo Datasets và Subsets
    raw_train = torchvision.datasets.FashionMNIST(
        root=str(data_dir), train=True, download=False, transform=train_tf
    )
    raw_val = torchvision.datasets.FashionMNIST(
        root=str(data_dir), train=True, download=False, transform=eval_tf
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=str(data_dir), train=False, download=False, transform=eval_tf
    )

    train_dataset = Subset(raw_train, split["train"])
    val_dataset = Subset(raw_val, split["val"])

    # 4. Tạo DataLoaders
    batch_size = loader_cfg.get("batch_size", 128)
    num_workers = loader_cfg.get("num_workers", 4)
    device = select_device()
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def to_sequence(images: Tensor, representation: Representation, patch_size: int = 4) -> Tensor:
    """Turn images (N, 1, 28, 28) into a sequence (N, T, F).

    rows     -> (N, 28, 28)
    columns  -> (N, 28, 28)
    patches  -> (N, (28 / patch_size) ** 2, patch_size ** 2)

    Issue #14.
    """
    n, c, h, w = images.shape
    assert c == 1 and h == 28 and w == 28, f"Expected shape (N, 1, 28, 28), got {images.shape}"

    if representation == "image":
        return images

    if representation == "rows":
        # Squeeze channel 1: (N, 1, 28, 28) -> (N, 28, 28)
        return images.squeeze(1)

    if representation == "columns":
        # (N, 1, 28, 28) -> squeeze -> (N, 28, 28) -> transpose H và W -> (N, 28, 28)
        return images.squeeze(1).transpose(1, 2).contiguous()

    if representation == "patches":
        p = patch_size
        assert h % p == 0 and w % p == 0, f"Image shape ({h}, {w}) not divisible by patch_size {p}"
        num_patches = (h // p) * (w // p)
        # Unfold spatial dimensions thành các patches
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(n, num_patches, p * p)
        return patches

    raise ValueError(f"Unknown representation: {representation}")


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