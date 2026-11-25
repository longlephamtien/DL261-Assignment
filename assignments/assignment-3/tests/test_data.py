"""The pairing, the split, and the batch contract the rest of the pipeline is written against."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.data import (
    MIRRORED, RefCOCOPairs, build_pairs, build_vocabulary, collate, group_by_object, load_tokenizer,
    verify_split,
)
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


class WordTokenizer:
    """Stands in for the Hugging Face tokenizer: one id per word, between [CLS] and [SEP]."""

    ids = {"[CLS]": 101, "[SEP]": 102, "man": 5, "left": 6, "right": 7}

    def __call__(self, text: str, truncation: bool = True) -> dict:
        return {"input_ids": [101, *(self.ids[word] for word in text.split()), 102]}


def two_objects() -> tuple[list[dict], dict]:
    refs = [
        {"ref_id": 1, "ann_id": 1, "image_id": 1, "split": "train", "category_id": 1,
         "sentences": [{"sent": "a", "tokens": ["man", "left"]}, {"sent": "b", "tokens": ["man", "right"]}]},
        {"ref_id": 2, "ann_id": 2, "image_id": 1, "split": "train", "category_id": 1,
         "sentences": [{"sent": "c", "tokens": ["man"]}]},
    ]
    return refs, {"images": [{"id": 1, "file_name": "x.jpg", "height": 4, "width": 4}], "annotations": []}


def test_the_eda_vocabulary_is_built_from_the_training_split_alone() -> None:
    """The EDA's out-of-vocabulary measurement must not count a test word as known."""
    refs = [
        {"split": "train", "sentences": [{"tokens": ["dog", "dog"]}]},
        {"split": "val", "sentences": [{"tokens": ["zebra", "zebra"]}]},
    ]
    vocabulary = build_vocabulary(refs, minimum=2)
    assert "dog" in vocabulary
    assert "zebra" not in vocabulary


def test_training_draws_one_expression_per_object_each_epoch() -> None:
    """The subset rule samples per epoch, so the second expression of an object is still seen."""
    refs, instances = two_objects()
    pairs = build_pairs(refs, instances, "train")
    assert [len(group) for group in group_by_object(pairs)] == [2, 1]

    training = RefCOCOPairs(pairs, instances, Path("."), WordTokenizer(), (4, 4), sample_per_object=True)
    evaluation = RefCOCOPairs(pairs, instances, Path("."), WordTokenizer(), (4, 4))
    assert len(training) == 2
    assert len(evaluation) == 3


def test_expressions_are_encoded_by_the_encoders_tokenizer() -> None:
    """The ids must index the pretrained embedding, with its own special tokens around them."""
    refs, instances = two_objects()
    dataset = RefCOCOPairs(build_pairs(refs, instances, "train"), instances, Path("."),
                           WordTokenizer(), (4, 4))
    assert dataset.encode(["man", "left"]) == [101, 5, 6, 102]


def test_the_tokenizer_pads_with_the_batch_contracts_id() -> None:
    """DistilBERT pads with 0, the id the collate function fills; a mismatch is refused."""
    path = Path.home() / ".cache/huggingface/hub/models--distilbert-base-uncased"
    if not path.is_dir():
        pytest.skip("distilbert-base-uncased is not cached")
    assert load_tokenizer("distilbert-base-uncased").pad_token_id == PAD_TOKEN


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
