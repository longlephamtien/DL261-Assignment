"""Cityscapes loading, the city-level split, and mask preprocessing.

    python -m src.data --config configs/base.yaml

Implemented by #60 and #61.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import requests
from cityscapesscripts.download.downloader import download_packages
from torch import Tensor
from torch.utils.data import DataLoader

from shared import env, io as shared_io

from .interfaces import SplitFile
from .utils import load_config, resolve_path

LOGIN_URL = "https://www.cityscapes-dataset.com/login"
PACKAGE_DIRS = {"leftImg8bit_trainvaltest": "leftImg8bit", "gtFine_trainvaltest": "gtFine"}
"""Top-level directory each archive unpacks into, used to skip work already done."""


def _session() -> requests.Session:
    """Log in with the account from `.env`, so the download needs no interactive prompt."""
    env.load()
    username, password = env.cityscapes()

    session = requests.Session()
    session.get(LOGIN_URL, allow_redirects=False).raise_for_status()
    response = session.post(
        LOGIN_URL,
        data={"username": username, "password": password, "submit": "Login"},
        allow_redirects=False,
    )
    response.raise_for_status()
    if response.status_code != 302:
        raise RuntimeError(f"Cityscapes rejected the credentials in .env for '{username}'")
    return session


def prepare_dataset(config: dict) -> Path:
    """Download the configured archives into `dataset.root` and unpack them; re-running is cheap."""
    dataset = config["dataset"]
    root = resolve_path(dataset["root"])
    root.mkdir(parents=True, exist_ok=True)

    pending = [name for name in dataset["packages"] if not (root / PACKAGE_DIRS[name]).is_dir()]
    archives = [f"{name}.zip" for name in pending if not (root / f"{name}.zip").is_file()]
    if archives:
        download_packages(session=_session(), package_names=archives, destination_path=str(root), resume=True)

    for name in pending:
        with zipfile.ZipFile(root / f"{name}.zip") as archive:
            archive.extractall(root)
    return root


def build_split(data_dir: Path, seed: int, val_cities: int) -> SplitFile:
    """Hold out whole cities from the official train split for model selection.

    The official validation split becomes our test set, because Cityscapes withholds the test
    labels. Selecting a checkpoint on the reported split would be selection on the test set,
    so the cities that choose it have to come out of train.
    """
    cities = sorted(p.name for p in (data_dir / "leftImg8bit" / "train").iterdir() if p.is_dir())
    if not 0 < val_cities < len(cities):
        raise ValueError(f"val_cities must be between 1 and {len(cities) - 1}, got {val_cities}")

    held_out = sorted(np.random.default_rng(seed).permutation(cities)[:val_cities])
    partitions = {
        "train": [city for city in cities if city not in held_out],
        "val": held_out,
        "test": sorted(p.name for p in (data_dir / "leftImg8bit" / "val").iterdir() if p.is_dir()),
    }

    split = SplitFile(
        dataset="cityscapes",
        seed=seed,
        split_unit="city",
        train=partitions["train"],
        val=partitions["val"],
        cities=partitions,
        counts={name: _count_images(data_dir, name, members) for name, members in partitions.items()},
    )
    verify_split(split)
    return split


def _count_images(data_dir: Path, partition: str, cities: list[str]) -> int:
    source = "val" if partition == "test" else "train"
    return sum(len(list((data_dir / "leftImg8bit" / source / city).glob("*.png"))) for city in cities)


def verify_split(split: SplitFile) -> None:
    """Fail if any city appears in more than one partition."""
    partitions = split["cities"]
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared_cities = set(partitions[left]) & set(partitions[right])
        if shared_cities:
            raise ValueError(f"{left} and {right} share {sorted(shared_cities)}; the split unit is the city")


def normalization_stats(data_dir: Path, split: SplitFile) -> tuple[list[float], list[float]]:
    """Per-channel mean and standard deviation over the training split only. Implemented by #60."""
    raise NotImplementedError("#60 implements the normalization statistics")


def to_train_ids(mask: Tensor) -> Tensor:
    """Cityscapes labelIds to the 19 trainIds plus IGNORE_INDEX. Implemented by #60.

    `csCreateTrainIdLabelImgs` from the official toolkit can do this offline instead.
    """
    raise NotImplementedError("#60 implements the label remapping")


def build_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Loaders for 'train', 'val', and 'test'. Implemented by #60."""
    raise NotImplementedError("#60 implements the data pipeline")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Cityscapes, then write the city-level split.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--download-only", action="store_true", help="fetch the archives and stop")
    args = parser.parse_args(argv)

    config = load_config(resolve_path(args.config))
    dataset = config["dataset"]
    root = prepare_dataset(config)
    print(root)
    if args.download_only:
        return 0

    split = build_split(root, config["seed"], dataset["val_cities"])
    shared_io.write_json(resolve_path(dataset["split_file"]), split)
    for partition, cities in split["cities"].items():
        print(f"{partition:<6} {len(cities):>2} cities  {split['counts'][partition]:>5} images  {', '.join(cities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
