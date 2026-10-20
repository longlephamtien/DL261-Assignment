"""Comparison contract: every run reaches the plot and the table, whatever the architecture."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.figures import COLOURS, configuration, default_name, label, plot_curves, summarize, variants
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


def test_every_configuration_is_plotted_beyond_the_palette(tmp_path: Path) -> None:
    """More configurations than colours; none may be dropped."""
    runs = {configuration(summary("m", 0, f"r{index}")): [history(0.8)] for index in range(len(COLOURS) + 3)}
    plot_curves(runs, tmp_path / "curves")
    assert (tmp_path / "curves.png").is_file()


def test_seeds_of_one_configuration_share_a_curve(tmp_path: Path) -> None:
    """Fifteen runs drawn individually are unreadable, so seeds average into one line."""
    runs = {"mlp/image": [history(0.88), history(0.89), history(0.90)]}
    plot_curves(runs, tmp_path / "curves")
    assert (tmp_path / "curves.png").is_file()


def test_curves_stop_at_the_shortest_seed(tmp_path: Path) -> None:
    """Early stopping gives seeds different lengths; the mean must not thin out at the tail."""
    runs = {"mlp/image": [history(0.88), history(0.89)[:1]]}
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


def config(**args: object) -> dict:
    return {"model": {"args": args}}


def test_variants_separate_cells_of_one_recurrent_configuration() -> None:
    """The cell ablation runs GRU and LSTM at the same model and representation."""
    summaries = [summary("recurrent", 0), summary("recurrent", 0)]
    configs = [config(cell_type="gru", representation="rows"), config(cell_type="lstm", representation="rows")]

    marks = variants(summaries, configs)
    assert marks == ["cell_type=gru", "cell_type=lstm"]
    assert label(summaries[0], marks[0]) != label(summaries[1], marks[1]), "runs must not collide"


def test_variants_stay_empty_when_a_model_appears_once() -> None:
    """A comparison across architectures is labelled by name, so nothing is appended."""
    summaries = [summary("recurrent", 0), summary("transformer", 0, "patches")]
    configs = [config(cell_type="gru", representation="rows"), config(d_model=128, representation="patches")]

    assert variants(summaries, configs) == ["", ""]
    assert configuration(summaries[0]) == "recurrent/image"


def test_variants_ignore_the_representation() -> None:
    """`representation` already keys the configuration and must not be repeated in the variant."""
    summaries = [summary("recurrent", 0, "rows"), summary("recurrent", 0, "patches")]
    configs = [config(cell_type="gru", representation="rows"), config(cell_type="gru", representation="patches")]

    assert variants(summaries, configs) == ["", ""]


def test_summarize_keeps_variants_apart() -> None:
    """Pooling GRU with LSTM would average two architectures into one row."""
    rows = [row("recurrent", 0, 0.87) | {"variant": "cell_type=gru"},
            row("recurrent", 0, 0.88) | {"variant": "cell_type=lstm"}]
    assert len(summarize(rows)) == 2
