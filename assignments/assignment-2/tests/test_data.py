"""The split contract, checked against a synthetic city tree rather than the real archives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.interfaces import CLASS_NAMES, IGNORE_INDEX
from src.data import (
    LABEL_TO_TRAIN_ID,
    CityscapesSegmentation,
    build_dataloaders,
    build_split,
    to_train_ids,
    verify_split,
)

TRAIN_CITIES = [f"city{index:02d}" for index in range(18)]
VAL_CITIES = ["frankfurt", "lindau", "munster"]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    for split, cities in (("train", TRAIN_CITIES), ("val", VAL_CITIES)):
        for city in cities:
            directory = tmp_path / "leftImg8bit" / split / city
            directory.mkdir(parents=True)
            for index in range(3):
                (directory / f"{city}_{index:06d}_leftImg8bit.png").touch()
    return tmp_path


def test_partitions_share_no_city(data_dir: Path) -> None:
    """The whole leakage argument rests on this; one shared city invalidates the comparison."""
    split = build_split(data_dir, seed=0, val_cities=3)
    partitions = split["cities"]
    assert set(partitions["train"]).isdisjoint(partitions["val"])
    assert set(partitions["train"]).isdisjoint(partitions["test"])
    assert set(partitions["val"]).isdisjoint(partitions["test"])


def test_selection_cities_come_out_of_train(data_dir: Path) -> None:
    """The official val split is reported, so the checkpoint cannot be chosen on it."""
    split = build_split(data_dir, seed=0, val_cities=3)
    assert len(split["cities"]["train"]) == len(TRAIN_CITIES) - 3
    assert set(split["cities"]["val"]) <= set(TRAIN_CITIES)
    assert split["cities"]["test"] == VAL_CITIES


def test_split_is_reproducible_and_seed_dependent(data_dir: Path) -> None:
    assert build_split(data_dir, seed=0, val_cities=3) == build_split(data_dir, seed=0, val_cities=3)
    assert build_split(data_dir, seed=0, val_cities=3)["val"] != build_split(data_dir, seed=1, val_cities=3)["val"]


def test_counts_match_the_files_on_disk(data_dir: Path) -> None:
    split = build_split(data_dir, seed=0, val_cities=3)
    assert split["counts"]["train"] == 3 * (len(TRAIN_CITIES) - 3)
    assert split["counts"]["test"] == 3 * len(VAL_CITIES)


def test_verify_split_rejects_an_overlap() -> None:
    bad = {"cities": {"train": ["aachen"], "val": ["aachen"], "test": ["frankfurt"]}}
    with pytest.raises(ValueError, match="share"):
        verify_split(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("val_cities", [0, 18, 99])
def test_val_cities_must_leave_a_training_set(data_dir: Path, val_cities: int) -> None:
    with pytest.raises(ValueError, match="val_cities"):
        build_split(data_dir, seed=0, val_cities=val_cities)


# --- label remapping, the piece every later metric depends on -------------------------------


def test_train_id_table_matches_the_official_cityscapes_definition() -> None:
    """Reference check: a transcription slip here would corrupt every mIoU silently."""
    from cityscapesscripts.helpers.labels import labels

    official = {label.id: label.trainId for label in labels if not label.ignoreInEval}
    assert LABEL_TO_TRAIN_ID == official


def test_train_id_names_line_up_with_the_class_list() -> None:
    """trainId i must be CLASS_NAMES[i], or the per-class report labels the wrong rows."""
    from cityscapesscripts.helpers.labels import labels

    official = {label.trainId: label.name for label in labels if not label.ignoreInEval}
    assert [official[index] for index in range(len(CLASS_NAMES))] == list(CLASS_NAMES)


def test_evaluated_labels_map_to_their_train_ids() -> None:
    mask = torch.tensor([[7, 8, 11], [26, 33, 23]], dtype=torch.uint8)
    assert to_train_ids(mask).tolist() == [[0, 1, 2], [13, 18, 10]]


def test_unevaluated_labels_become_ignore_index() -> None:
    """Void, ego vehicle, parking, rail track and guard rail carry no evaluated label."""
    mask = torch.tensor([0, 1, 9, 10, 14, 255], dtype=torch.uint8)
    assert to_train_ids(mask).tolist() == [IGNORE_INDEX] * 6


def test_every_train_id_is_reachable() -> None:
    """All 19 classes must be producible, or one of them could never be predicted correctly."""
    mask = torch.tensor(sorted(LABEL_TO_TRAIN_ID), dtype=torch.uint8)
    assert sorted(to_train_ids(mask).tolist()) == list(range(len(CLASS_NAMES)))


def test_remapping_preserves_shape() -> None:
    assert to_train_ids(torch.zeros(4, 8, dtype=torch.uint8)).shape == (4, 8)


ROAD_LABEL_ID, SIDEWALK_LABEL_ID = 7, 8
ROAD_TRAIN_ID, SIDEWALK_TRAIN_ID = 0, 1


def write_pair(root: Path, source: str, city: str) -> None:
    """One image and mask whose left half is road and right half sidewalk, so a flip is visible."""
    image_dir, mask_dir = root / "leftImg8bit" / source / city, root / "gtFine" / source / city
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    mask = np.full((4, 8), ROAD_LABEL_ID, dtype=np.uint8)
    mask[:, 4:] = SIDEWALK_LABEL_ID
    Image.fromarray(np.stack([mask * 10] * 3, axis=-1)).save(image_dir / f"{city}_000000_leftImg8bit.png")
    Image.fromarray(mask).save(mask_dir / f"{city}_000000_gtFine_labelIds.png")


def flipped(dataset: CityscapesSegmentation) -> bool:
    _, mask = dataset[0]
    return mask[0, 0].item() == SIDEWALK_TRAIN_ID


@pytest.fixture
def image_tree(tmp_path: Path) -> Path:
    write_pair(tmp_path, "train", "aachen")
    return tmp_path


def augmented(root: Path, **kwargs) -> CityscapesSegmentation:
    return CityscapesSegmentation(root, ["aachen"], "train", size=(4, 8), augment=True, **kwargs)


def test_flip_probability_zero_never_flips(image_tree: Path) -> None:
    dataset = augmented(image_tree, flip_probability=0.0)
    assert not any(flipped(dataset) for _ in range(20))


def test_flip_probability_one_always_flips_image_and_mask_together(image_tree: Path) -> None:
    plain = augmented(image_tree, flip_probability=0.0)
    mirrored = augmented(image_tree, flip_probability=1.0)

    image, mask = plain[0]
    flipped_image, flipped_mask = mirrored[0]

    assert torch.equal(flipped_image, image.flip(-1))
    assert torch.equal(flipped_mask, mask.flip(-1))


def test_the_default_flip_probability_flips_some_samples_but_not_all(image_tree: Path) -> None:
    torch.manual_seed(0)
    dataset = augmented(image_tree)
    outcomes = {flipped(dataset) for _ in range(50)}
    assert outcomes == {True, False}


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_a_flip_probability_outside_the_unit_interval_is_rejected(image_tree: Path, probability: float) -> None:
    with pytest.raises(ValueError, match="flip_probability"):
        augmented(image_tree, flip_probability=probability)


def test_the_same_seed_reproduces_the_same_augmentation(image_tree: Path) -> None:
    dataset = CityscapesSegmentation(image_tree, ["aachen"], "train", size=(4, 8), crop=(2, 3), augment=True)

    def draw(seed: int) -> list[torch.Tensor]:
        torch.manual_seed(seed)
        return [dataset[0][1] for _ in range(20)]

    assert all(torch.equal(left, right) for left, right in zip(draw(0), draw(0)))
    assert not all(torch.equal(left, right) for left, right in zip(draw(0), draw(1)))


@pytest.fixture
def loader_config(tmp_path: Path) -> dict:
    write_pair(tmp_path, "train", "aachen")
    write_pair(tmp_path, "train", "bochum")
    write_pair(tmp_path, "val", "lindau")
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"cities": {"train": ["aachen"], "val": ["bochum"], "test": ["lindau"]}}))
    return {
        "dataset": {"root": str(tmp_path), "split_file": str(split_file)},
        "input": {"height": 4, "width": 8, "crop": [2, 2]},
        "loader": {"batch_size": 1, "num_workers": 0, "augment": True, "flip_probability": 0.25},
    }


def test_the_configured_flip_probability_reaches_the_training_loader(loader_config: dict) -> None:
    assert build_dataloaders(loader_config)["train"].dataset.flip_probability == 0.25


def test_a_run_recorded_before_the_flip_key_existed_keeps_its_probability(loader_config: dict) -> None:
    """`src.evaluate` replays each run's own config.yaml, including the published checkpoints."""
    del loader_config["loader"]["flip_probability"]
    assert build_dataloaders(loader_config)["train"].dataset.flip_probability == 0.5
