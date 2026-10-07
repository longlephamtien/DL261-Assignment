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
