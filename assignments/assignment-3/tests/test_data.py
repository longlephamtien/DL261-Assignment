"""The pairing, the split, and the batch contract the rest of the pipeline is written against."""

from __future__ import annotations

import json

import pytest
import torch

from src.data import MIRRORED, build_pairs, build_vocabulary, collate, verify_split
from src.interfaces import PAD_TOKEN, SPLITS
from src.utils import resolve_path


@pytest.fixture(scope="module")
def split() -> dict:
    path = resolve_path("splits/refcoco-unc.json")
    if not path.is_file():
        pytest.skip("run python -m src.data --split first")
    return json.loads(path.read_text())


def test_no_image_appears_in_two_splits(split: dict) -> None:
    """The leakage guarantee the proposal rests on, checked against the committed file."""
    seen: dict[int, str] = {}
    for name, ids in split["images"].items():
        for image_id in ids:
            assert image_id not in seen, f"image {image_id} in {seen.get(image_id)} and {name}"
            seen[image_id] = name


def test_split_counts_cover_every_partition(split: dict) -> None:
    assert set(split["counts"]) == set(SPLITS)
    assert all(split["counts"][name]["images"] == len(split["images"][name]) for name in SPLITS)


def test_verify_split_rejects_an_overlapping_split(split: dict) -> None:
    """A guard that never fires is a guard that is not tested."""
    broken = {**split, "images": {**split["images"], "val": split["images"]["val"] + [split["images"]["train"][0]]}}
    with pytest.raises(ValueError, match="both"):
        verify_split(broken, [])


def test_build_pairs_applies_the_subset_rule() -> None:
    """One expression per object during training, every expression at evaluation."""
    refs = [{"ref_id": 1, "ann_id": 1, "image_id": 1, "split": "train", "category_id": 1,
             "sentences": [{"sent": "a", "tokens": ["a"]}, {"sent": "b", "tokens": ["b"]}]}]
    instances = {"images": [{"id": 1, "file_name": "x.jpg"}]}
    assert len(build_pairs(refs, instances, "train", per_object=1)) == 1
    assert len(build_pairs(refs, instances, "train", per_object=None)) == 2


def test_vocabulary_is_built_from_the_training_split_alone() -> None:
    """A word seen only at test time must arrive as <unk>, not as a known id."""
    refs = [
        {"split": "train", "sentences": [{"tokens": ["dog", "dog"]}]},
        {"split": "val", "sentences": [{"tokens": ["zebra", "zebra"]}]},
    ]
    vocabulary = build_vocabulary(refs, minimum=2)
    assert "dog" in vocabulary
    assert "zebra" not in vocabulary


def test_collate_pads_to_the_longest_and_marks_real_tokens() -> None:
    samples = [
        {"image": torch.zeros(3, 4, 4), "tokens": torch.tensor([5, 6, 7]),
         "mask": torch.zeros(4, 4, dtype=torch.long), "ref_id": 1},
        {"image": torch.zeros(3, 4, 4), "tokens": torch.tensor([8]),
         "mask": torch.zeros(4, 4, dtype=torch.long), "ref_id": 2},
    ]
    batch = collate(samples)
    assert batch["tokens"].shape == (2, 3)
    assert batch["tokens"][1].tolist() == [8, PAD_TOKEN, PAD_TOKEN]
    assert batch["attention"][1].tolist() == [True, False, False]
    assert batch["images"].shape[0] == batch["masks"].shape[0] == 2


def test_a_horizontal_flip_rewrites_the_words_it_inverts() -> None:
    """46.3% of expressions name a side, so flipping pixels alone would relabel them."""
    assert MIRRORED["left"] == "right" and MIRRORED["right"] == "left"
    assert [MIRRORED.get(t, t) for t in ["the", "left", "dog"]] == ["the", "right", "dog"]
