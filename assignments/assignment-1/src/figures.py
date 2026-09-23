"""Plots and tables derived from finished runs.

`train.fit` calls `plot_curves` itself, so every run directory already holds its own curves.
This module compares runs against each other:

    python -m src.figures results/runs/<run_id> [...]

writing the combined curves and the validation metrics table under `results/figures/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from shared import io as shared_io

from .interfaces import EpochRecord, RunSummary
from .utils import resolve_path

FIGURES = "results/figures"
COLOURS = ("#1488D8", "#030391", "#C1440E", "#2E7D32", "#6A1B9A")


def load_run(run_dir: Path) -> tuple[RunSummary, list[EpochRecord]]:
    return shared_io.read_json(run_dir / "summary.json"), shared_io.read_json(run_dir / "history.json")


def plot_curves(runs: dict[str, list[EpochRecord]], path: Path) -> None:
    """Training and validation loss and macro-F1 for every run, on shared axes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for (name, history), colour in zip(runs.items(), COLOURS):
        epochs = [record["epoch"] for record in history]
        axes[0].plot(epochs, [r["train_loss"] for r in history], color=colour, label=f"{name} train")
        axes[0].plot(epochs, [r["val_loss"] for r in history], color=colour, linestyle="--", label=f"{name} val")
        axes[1].plot(epochs, [r["train_macro_f1"] for r in history], color=colour, label=f"{name} train")
        axes[1].plot(epochs, [r["val_macro_f1"] for r in history], color=colour, linestyle="--", label=f"{name} val")

    axes[0].set_ylabel("cross-entropy loss")
    axes[1].set_ylabel("macro-F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Learning curves, solid train and dashed validation")
    save(fig, path)


def save(fig: plt.Figure, path: Path) -> None:
    """PNG for the repository, SVG and PDF for the site and the report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    for suffix in (".svg", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export learning curves and the metrics table for finished runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="paths under results/runs/")
    args = parser.parse_args(argv)

    histories: dict[str, list[EpochRecord]] = {}
    rows = []
    for run_dir in args.run_dirs:
        summary, history = load_run(resolve_path(run_dir))
        histories[summary["model"]] = history
        best = next(record for record in history if record["epoch"] == summary["best_epoch"])
        environment = shared_io.read_json(resolve_path(run_dir) / "environment.json")
        rows.append(
            {
                "model": summary["model"],
                "parameters": summary["parameters"],
                "best_epoch": summary["best_epoch"],
                "epochs_trained": summary["epochs_trained"],
                "training_seconds": round(summary["training_seconds"], 1),
                "seconds_per_epoch": round(sum(r["seconds"] for r in history) / len(history), 1),
                "val_accuracy": round(best["val_accuracy"], 4),
                "val_macro_f1": round(best["val_macro_f1"], 4),
                "hardware": summary["hardware"],
                "torch_threads": environment.get("torch_threads"),
            }
        )

    figures = resolve_path(FIGURES)
    plot_curves(histories, figures / "learning_curves")
    shared_io.write_json(figures / "validation_metrics.json", rows)

    print(f"{'model':<8}{'params':>10}{'best':>6}{'epochs':>8}{'seconds':>10}{'s/epoch':>9}{'val acc':>10}{'val F1':>9}")
    for row in rows:
        print(
            f"{row['model']:<8}{row['parameters']:>10,}{row['best_epoch']:>6}{row['epochs_trained']:>8}"
            f"{row['training_seconds']:>10.1f}{row['seconds_per_epoch']:>9.1f}"
            f"{row['val_accuracy']:>10.4f}{row['val_macro_f1']:>9.4f}"
        )

    devices = {row["hardware"] for row in rows}
    if len(devices) > 1:
        print(f"\nwarning: runs come from different machines, so the times are not comparable: {devices}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
