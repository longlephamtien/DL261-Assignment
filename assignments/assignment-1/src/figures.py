"""Plots and tables derived from finished runs.

    python -m src.figures <run_id> [...] [--name baselines] [--split val]
"""

from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

from shared import io as shared_io

from .interfaces import EVALUATION_SPLITS, EpochRecord, EvaluationReport, RunSummary, Split
from .utils import report_path, resolve_path, run_dir as resolve_run

FIGURES = "results/figures"
COLOURS = ("#1488D8", "#030391", "#C1440E", "#2E7D32", "#6A1B9A")


def load_run(run_dir: Path) -> tuple[dict, RunSummary, list[EpochRecord]]:
    return (
        shared_io.read_yaml(run_dir / "config.yaml"),
        shared_io.read_json(run_dir / "summary.json"),
        shared_io.read_json(run_dir / "history.json"),
    )


def label(summary: RunSummary) -> str:
    """Curve label that stays unique across seeds and across sequence representations."""
    return f"{summary['model']}/{summary['representation']} seed{summary['seed']}"


def default_name(summaries: list[RunSummary]) -> str:
    """Name a comparison after the models it covers."""
    return "_".join(sorted({summary["model"] for summary in summaries}))


def plot_curves(runs: dict[str, list[EpochRecord]], path: Path) -> None:
    """Training and validation loss and macro-F1 for every run, on shared axes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for (name, history), colour in zip(runs.items(), cycle(COLOURS)):
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


def describe(summary: RunSummary, history: list[EpochRecord], report: EvaluationReport, environment: dict) -> dict:
    """One table row, unrounded so `summarize` does not take a deviation of rounded numbers."""
    return {
        "run_id": summary["run_id"],
        "model": summary["model"],
        "representation": summary["representation"],
        "seed": summary["seed"],
        "parameters": report["parameters"],
        "best_epoch": summary["best_epoch"],
        "epochs_trained": summary["epochs_trained"],
        "training_seconds": summary["training_seconds"],
        "seconds_per_epoch": sum(r["seconds"] for r in history) / len(history),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "inference_ms_per_image": report["inference_ms_per_image"],
        "hardware": summary["hardware"],
        "torch_threads": environment.get("torch_threads"),
    }


def summarize(rows: list[dict]) -> list[dict]:
    """Mean and standard deviation of macro-F1 per configuration, across its seeds."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["representation"]), []).append(row)

    configurations = []
    for (model, representation), members in groups.items():
        scores = [row["macro_f1"] for row in members]
        configurations.append(
            {
                "model": model,
                "representation": representation,
                "seeds": sorted(row["seed"] for row in members),
                "parameters": members[0]["parameters"],
                "macro_f1_mean": mean(scores),
                "macro_f1_sd": stdev(scores) if len(scores) > 1 else 0.0,
                "accuracy_mean": mean(row["accuracy"] for row in members),
            }
        )
    return configurations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export learning curves and the metrics table for finished runs.")
    parser.add_argument("runs", nargs="+", type=resolve_run, help="run ids, or paths to the run directories")
    parser.add_argument("--name", default=None, help="comparison directory; defaults to the models compared")
    parser.add_argument("--split", choices=EVALUATION_SPLITS, default="val", help="which reports to tabulate")
    args = parser.parse_args(argv)
    split: Split = args.split

    histories: dict[str, list[EpochRecord]] = {}
    summaries: list[RunSummary] = []
    rows = []
    for run_dir in args.runs:
        config, summary, history = load_run(run_dir)
        report = report_path(config, summary["run_id"], split)
        if not report.is_file():
            parser.error(f"no {split} report for {summary['run_id']}; run python -m src.evaluate {summary['run_id']} --split {split}")

        key = label(summary)
        if key in histories:
            parser.error(f"two runs given for {key}; a comparison takes one run per configuration and seed")

        environment = shared_io.read_json(run_dir / "environment.json")
        rows.append(describe(summary, history, shared_io.read_json(report), environment))
        histories[key] = history
        summaries.append(summary)

    configurations = summarize(rows)
    comparison = resolve_path(FIGURES) / (args.name or default_name(summaries))
    plot_curves(histories, comparison / "learning_curves")
    shared_io.write_json(
        comparison / f"metrics_{split}.json", {"split": split, "runs": rows, "configurations": configurations}
    )

    print(f"{'model':<12}{'repr':<11}{'seed':>5}{'params':>10}{'best':>6}{'s/epoch':>9}{'accuracy':>10}{'macro-F1':>10}")
    for row in rows:
        print(
            f"{row['model']:<12}{row['representation']:<11}{row['seed']:>5}{row['parameters']:>10,}"
            f"{row['best_epoch']:>6}{row['seconds_per_epoch']:>9.1f}{row['accuracy']:>10.4f}{row['macro_f1']:>10.4f}"
        )

    print(f"\n{'model':<12}{'repr':<11}{'seeds':>6}{'macro-F1 mean':>15}{'sd':>8}")
    for group in configurations:
        print(
            f"{group['model']:<12}{group['representation']:<11}{len(group['seeds']):>6}"
            f"{group['macro_f1_mean']:>15.4f}{group['macro_f1_sd']:>8.4f}"
        )

    devices = {row["hardware"] for row in rows}
    if len(devices) > 1:
        print(f"\nwarning: runs come from different machines, so the times are not comparable: {devices}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
