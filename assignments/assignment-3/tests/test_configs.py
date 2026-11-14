"""Decisions the proposal committed to, which a later config edit could undo silently."""

from __future__ import annotations

from src.utils import load_config, resolve_path


def config() -> dict:
    return load_config(resolve_path("configs/base.yaml"))


def test_partition_stays_unc() -> None:
    """The Google partition puts 88.7% of its validation images in train as well."""
    assert config()["dataset"]["partition"] == "unc"


def test_annotation_checksum_is_pinned() -> None:
    """The source is an archived mirror, so the checksum is what makes it trustworthy."""
    assert len(config()["dataset"]["annotations_sha256"]) == 64


def test_training_keeps_at_most_one_expression_per_object() -> None:
    """The subset rule the proposal declared; evaluation always reads every expression."""
    assert config()["loader"]["expressions_per_object"] == 1


def test_every_model_config_leaves_the_shared_protocol_alone() -> None:
    """The protocol is fixed in base.yaml; a model file varies only its own block."""
    base = config()["train"]
    for name in ("image-only", "text-only", "simple-fusion", "improved-fusion"):
        assert load_config(resolve_path(f"configs/{name}.yaml"))["train"] == base


def test_every_model_config_names_a_registered_model() -> None:
    from src.models import list_models

    registered = set(list_models())
    for name in ("image-only", "text-only", "simple-fusion", "improved-fusion"):
        assert load_config(resolve_path(f"configs/{name}.yaml"))["model"]["name"] in registered


def test_the_two_fusion_arms_differ_only_in_the_fusion() -> None:
    """The ablation isolates the fusion, so nothing else may drift between them."""
    simple = load_config(resolve_path("configs/simple-fusion.yaml"))["model"]["args"]
    improved = load_config(resolve_path("configs/improved-fusion.yaml"))["model"]["args"]
    assert simple["backbone"] == improved["backbone"]
    assert simple["encoder"] == improved["encoder"]
    assert simple["fusion"] != improved["fusion"]
