"""The split contract, checked against a synthetic city tree rather than the real archives."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data import build_split, verify_split

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
