"""Cityscapes loading, the city-level split, and mask preprocessing.

    python -m src.data --config configs/base.yaml

Implemented by #60 and #61.
"""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
from cityscapesscripts.download.downloader import download_packages
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from shared import env, io as shared_io

from .interfaces import IGNORE_INDEX, SplitFile
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
    """Hold out whole cities from the official train split for model selection."""
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


IMAGENET_NORMALIZATION = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
"""The ImageNet channel statistics the pretrained torchvision weights were trained with."""


def _sample_paths(data_dir: Path, partition: str, cities: list[str]) -> tuple[list[tuple[Path, Path]], int]:
    """Pair every image with its labelIds mask, returning the pairs and the count of unusable ones.

    Only paths are held, never pixels: the full set is 5 GB of PNG, so the index must stay small
    and each worker decodes its own images on demand.
    """
    source = "val" if partition == "test" else "train"
    pairs: list[tuple[Path, Path]] = []
    invalid = 0

    for city in sorted(cities):
        for image in sorted((data_dir / "leftImg8bit" / source / city).glob("*_leftImg8bit.png")):
            mask = data_dir / "gtFine" / source / city / image.name.replace(
                "_leftImg8bit.png", "_gtFine_labelIds.png"
            )
            if mask.is_file():
                pairs.append((image, mask))
            else:
                invalid += 1
    return pairs, invalid


class CityscapesSegmentation(Dataset):
    """Images and masks for one partition, decoded per item rather than held in memory."""

    def __init__(
        self,
        data_dir: Path,
        cities: list[str],
        partition: str,
        size: tuple[int, int],
        crop: tuple[int, int] | None = None,
        augment: bool = False,
        normalization: tuple[list[float], list[float]] = IMAGENET_NORMALIZATION,
    ) -> None:
        self.samples, self.invalid_annotations = _sample_paths(data_dir, partition, cities)
        if not self.samples:
            raise ValueError(f"no image and mask pairs for {partition} in {data_dir}")
        self.size, self.crop, self.augment = size, crop, augment
        mean, std = normalization
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path, mask_path = self.samples[index]
        height, width = self.size

        image = Image.open(image_path).convert("RGB").resize((width, height), Image.BILINEAR)
        mask = Image.open(mask_path).resize((width, height), Image.NEAREST)

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask = to_train_ids(torch.from_numpy(np.array(mask)))

        if self.crop is not None:
            image, mask = self._crop(image, mask)
        if self.augment and torch.rand(()) < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)

        return (image - self.mean) / self.std, mask.long()

    def _crop(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Random window while training, centre window otherwise, so validation is deterministic."""
        crop_height, crop_width = self.crop
        _, height, width = image.shape
        if self.augment:
            top = int(torch.randint(0, max(height - crop_height, 0) + 1, ()))
            left = int(torch.randint(0, max(width - crop_width, 0) + 1, ()))
        else:
            top, left = max(height - crop_height, 0) // 2, max(width - crop_width, 0) // 2
        return (
            image[:, top : top + crop_height, left : left + crop_width],
            mask[top : top + crop_height, left : left + crop_width],
        )


def normalization_stats(data_dir: Path, split: SplitFile, limit: int | None = None) -> tuple[list[float], list[float]]:
    """Per-channel mean and standard deviation over the training split only.

    Accumulated as running sums of x and x squared, so the pass costs one image of memory
    rather than the whole split. `limit` samples a prefix when a quick estimate is enough.
    """
    dataset = CityscapesSegmentation(
        data_dir, split["cities"]["train"], "train", size=(512, 1024),
        normalization=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),  # raw [0, 1] pixels
    )
    count = len(dataset) if limit is None else min(limit, len(dataset))

    total = torch.zeros(3, dtype=torch.float64)
    total_squared = torch.zeros(3, dtype=torch.float64)
    pixels = 0
    for index in range(count):
        image, _ = dataset[index]
        flat = image.reshape(3, -1).to(torch.float64)
        total += flat.sum(dim=1)
        total_squared += (flat**2).sum(dim=1)
        pixels += flat.shape[1]

    mean = total / pixels
    std = (total_squared / pixels - mean**2).clamp_min(0).sqrt()
    return mean.tolist(), std.tolist()

LABEL_TO_TRAIN_ID = {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}

MAX_LABEL_ID = 33
"""Highest labelId Cityscapes assigns. -1 marks a license plate and is also ignored."""


def _lookup_table() -> Tensor:
    """A 256-entry table so the remapping is one indexing operation per mask."""
    table = torch.full((256,), IGNORE_INDEX, dtype=torch.uint8)
    for label_id, train_id in LABEL_TO_TRAIN_ID.items():
        table[label_id] = train_id
    return table


_TRAIN_ID_TABLE = _lookup_table()


def to_train_ids(mask: Tensor) -> Tensor:
    """Cityscapes labelIds to the 19 trainIds plus IGNORE_INDEX.

    `csCreateTrainIdLabelImgs` from the official toolkit can do this offline instead. Doing it
    here keeps the downloaded archives untouched, so the data directory stays reproducible from
    the published packages alone.

    Every labelId outside the evaluated 19 becomes IGNORE_INDEX rather than a class, which is
    what makes the 11.8 percent of ignored pixels measured in the EDA invisible to the metrics.
    """
    if mask.dtype != torch.uint8:
        mask = mask.to(torch.uint8)
    return _TRAIN_ID_TABLE.to(mask.device)[mask.long()]


def build_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Loaders for 'train', 'val', and 'test'.

    Only the training loader crops and augments. Validation and test see the whole resized
    image, so a score is never an average over arbitrary windows.
    """
    dataset_config = config["dataset"]
    data_dir = resolve_path(dataset_config["root"])
    split: SplitFile = shared_io.read_json(resolve_path(dataset_config["split_file"]))
    verify_split(split)

    size = (config["input"]["height"], config["input"]["width"])
    crop = tuple(config["input"]["crop"])
    loader_config = config["loader"]
    normalization = tuple(dataset_config.get("normalization", IMAGENET_NORMALIZATION))

    loaders = {}
    for partition in ("train", "val", "test"):
        training = partition == "train"
        dataset = CityscapesSegmentation(
            data_dir,
            split["cities"][partition],
            partition,
            size=size,
            crop=crop if training else None,
            augment=training and loader_config["augment"],
            normalization=normalization,
        )
        if dataset.invalid_annotations:
            print(f"{partition}: skipped {dataset.invalid_annotations} images with no labelIds mask")
        loaders[partition] = DataLoader(
            dataset,
            batch_size=loader_config["batch_size"],
            shuffle=training,
            num_workers=loader_config["num_workers"],
            pin_memory=True,
            drop_last=training,
            persistent_workers=loader_config["num_workers"] > 0,
        )
    return loaders


def measure_throughput(loader: DataLoader, batches: int = 20) -> dict[str, float]:
    """Images per second the loader sustains, after a warm-up batch that pays worker start-up."""
    iterator = iter(loader)
    first = next(iterator)
    images_per_batch = first[0].shape[0]

    start = time.perf_counter()
    seen = 0
    for _ in range(batches):
        try:
            next(iterator)
        except StopIteration:
            break
        seen += 1
    elapsed = time.perf_counter() - start

    return {
        "batches": seen,
        "images": seen * images_per_batch,
        "seconds": elapsed,
        "images_per_second": seen * images_per_batch / elapsed if elapsed > 0 else 0.0,
    }


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
