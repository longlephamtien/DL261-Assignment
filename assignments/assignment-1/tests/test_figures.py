"""Comparison contract: every run reaches the plot and the table, whatever the architecture."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.figures import COLOURS, default_name, label, plot_curves, summarize
from src.interfaces import RunSummary


def summary(model: str, seed: int, representation: str = "image") -> RunSummary:
    return {"model": model, "seed": seed, "representation": representation}  # type: ignore[typeddict-item]


def history(macro_f1: float) -> list[dict]:
    return [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "train_macro_f1": 0.5, "val_macro_f1": macro_f1},
        {"epoch": 2, "train_loss": 0.8, "val_loss": 0.7, "train_macro_f1": 0.6, "val_macro_f1": macro_f1},
    ]


def row(model: str, seed: int, macro_f1: float, representation: str = "image") -> dict:
    return {
        "model": model,
        "representation": representation,
        "seed": seed,
        "parameters": 1000,
        "macro_f1": macro_f1,
        "accuracy": macro_f1,
    }


def test_label_separates_seeds_and_representations() -> None:
    """The protocol runs one model at several seeds and several representations."""
    labels = {
        label(summary("transformer", 0, "patches")),
        label(summary("transformer", 1, "patches")),
        label(summary("transformer", 0, "rows")),
    }
    assert len(labels) == 3


def test_every_run_is_plotted_beyond_the_palette(tmp_path: Path) -> None:
    """Five models at three seeds exceed the colour list; none may be dropped."""
    runs = {label(summary("m", seed, f"r{index}")): history(0.8) for index, seed in enumerate(range(len(COLOURS) + 3))}
    plot_curves(runs, tmp_path / "curves")
    assert (tmp_path / "curves.png").is_file()


def test_summarize_groups_seeds_of_one_configuration() -> None:
    rows = [row("mlp", 0, 0.88), row("mlp", 1, 0.90), row("linear", 0, 0.77)]
    configurations = {group["model"]: group for group in summarize(rows)}

    assert configurations["mlp"]["seeds"] == [0, 1]
    assert configurations["mlp"]["macro_f1_mean"] == pytest.approx(0.89)
    assert configurations["mlp"]["macro_f1_sd"] > 0
    assert configurations["linear"]["macro_f1_sd"] == 0.0, "one seed measures no spread"


def test_default_name_covers_the_models_compared() -> None:
    """Two comparisons over different model sets must not land in the same directory."""
    baselines = [summary("linear", 0), summary("mlp", 0)]
    architectures = baselines + [summary("transformer", 0, "patches")]
    assert default_name(baselines) != default_name(architectures)
    assert default_name(baselines) == default_name(list(reversed(baselines))), "order must not matter"


def test_summarize_keeps_representations_apart() -> None:
    """E2 compares one architecture across representations, so they cannot be pooled."""
    rows = [row("transformer", 0, 0.90, "patches"), row("transformer", 0, 0.85, "rows")]
    assert len(summarize(rows)) == 2
