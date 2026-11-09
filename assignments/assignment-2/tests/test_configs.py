"""Decisions the proposal committed to, which a later config edit could undo silently."""

from __future__ import annotations

import pytest

from src.utils import load_config, resolve_path


@pytest.mark.parametrize("model", ["unet", "deeplabv3"])
def test_model_configs_leave_the_shared_training_protocol_alone(model: str) -> None:
    """Every model trains under one protocol; tuning varies `model.args` only."""
    base = load_config(resolve_path("configs/base.yaml"))["train"]
    assert load_config(resolve_path(f"configs/{model}.yaml"))["train"] == base


def test_split_unit_stays_the_approved_one() -> None:
    """Splitting by anything finer than the city puts one street scene in two partitions."""
    assert load_config(resolve_path("configs/base.yaml"))["dataset"]["split_unit"] == "city"


def test_fine_tuning_procedure_stays_the_committed_one() -> None:
    """The single-stage procedure from the proposal; an experiment varies these through `--set`."""
    config = load_config(resolve_path("configs/deeplabv3.yaml"))
    train = config["train"]

    assert config["model"]["args"]["pretrained"] is True
    assert config["model"]["args"]["freeze_backbone"] is False
    assert (train["optimizer"], train["lr"], train["weight_decay"]) == ("adamw", 3e-4, 0.05)
    assert (train["scheduler"], train["epochs"], train["early_stopping_patience"]) == ("cosine", 40, 8)
    assert (train["checkpoint_metric"], train["checkpoint_mode"]) == ("val_miou", "max")
    assert config["loader"]["augment"] is True
    assert config["loader"]["flip_probability"] == 0.5
    assert config["input"]["crop"] == [512, 512]


def test_an_experiment_can_override_the_augmentation_from_the_command_line() -> None:
    config = load_config(resolve_path("configs/deeplabv3.yaml"), {"loader.flip_probability": "0.25"})
    assert config["loader"]["flip_probability"] == 0.25
