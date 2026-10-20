"""Plots and tables derived from finished runs.

    python -m src.figures <run_id> [...] [--name baselines] [--split val]
"""

from __future__ import annotations

import argparse
import re
from itertools import cycle
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

from shared import io as shared_io
from shared.figures import COLOURS, save

from .interfaces import CLASS_NAMES, EVALUATION_SPLITS, EpochRecord, EvaluationReport, RunSummary, Split
from .utils import report_path, resolve_path, run_dir as resolve_run

FIGURES = "results/figures"


def load_run(run_dir: Path) -> tuple[dict, RunSummary, list[EpochRecord]]:
    return (
        shared_io.read_yaml(run_dir / "config.yaml"),
        shared_io.read_json(run_dir / "summary.json"),
        shared_io.read_json(run_dir / "history.json"),
    )


def variants(summaries: list[RunSummary], configs: list[dict]) -> list[str]:
    """Model arguments that distinguish runs of the same model, such as the recurrent cell.

    Compared within a model name, so a comparison across architectures stays unlabelled;
    `representation` is excluded because it already keys the configuration.
    """
    groups: dict[str, list[int]] = {}
    for index, summary in enumerate(summaries):
        groups.setdefault(summary["model"], []).append(index)

    labels = [""] * len(summaries)
    for indices in groups.values():
        args = [configs[index].get("model", {}).get("args", {}) for index in indices]
        keys = {key for arg in args for key in arg} - {"representation", "patch_size"}
        varying = sorted(key for key in keys if len({repr(arg.get(key)) for arg in args}) > 1)
        for index, arg in zip(indices, args):
            labels[index] = ",".join(f"{key}={arg.get(key)}" for key in varying)
    return labels


def configuration(summary: RunSummary, variant: str = "") -> str:
    """Curves are grouped by configuration, not by run; seeds of one configuration average."""
    return f"{summary['model']}/{summary['representation']}" + (f"/{variant}" if variant else "")


def label(summary: RunSummary, variant: str = "") -> str:
    """Table row label, unique across seeds, representations and model variants."""
    return f"{configuration(summary, variant)} seed{summary['seed']}"


def default_name(summaries: list[RunSummary]) -> str:
    """Name a comparison after the models it covers."""
    return "_".join(sorted({summary["model"] for summary in summaries}))


def _band(histories: list[list[EpochRecord]], key: str) -> tuple[list[int], list[float], list[float]]:
    """Mean and standard deviation of one metric across seeds, epoch by epoch."""
    length = min(len(h) for h in histories)
    epochs = [histories[0][i]["epoch"] for i in range(length)]
    values = [[h[i][key] for h in histories] for i in range(length)]
    means = [mean(v) for v in values]
    deviations = [stdev(v) if len(v) > 1 else 0.0 for v in values]
    return epochs, means, deviations


def plot_curves(runs: dict[str, list[list[EpochRecord]]], path: Path) -> None:
    """Training and validation curves per configuration, averaged over its seeds."""
    panels = (("loss", "cross-entropy loss"), ("macro_f1", "macro-F1"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    for (name, histories), colour in zip(runs.items(), cycle(COLOURS)):
        seeds = len(histories)
        for ax, (key, _) in zip(axes, panels):
            for split, style, width in (("val", "-", 1.8), ("train", "--", 1.0)):
                epochs, means, deviations = _band(histories, f"{split}_{key}")
                ax.plot(epochs, means, color=colour, linestyle=style, linewidth=width,
                        label=name if split == "val" else None)
                if seeds > 1:
                    ax.fill_between(epochs,
                                    [m - d for m, d in zip(means, deviations)],
                                    [m + d for m, d in zip(means, deviations)],
                                    color=colour, alpha=0.15, linewidth=0)

    for ax, (_, ylabel) in zip(axes, panels):
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")

    seeds = {len(h) for h in runs.values()}
    spread = f", shaded band one standard deviation over {seeds.pop()} seeds" if len(seeds) == 1 else ""
    fig.suptitle(f"Learning curves, solid validation and dashed training{spread}")
    save(fig, path)


def plot_confusion(report: EvaluationReport, path: Path) -> None:
    """Row-normalised confusion matrix: what each true class was predicted as.

    Rows are normalised because the split is balanced per class but the errors are not, so
    raw counts hide which classes a model actually confuses.
    """
    counts = np.array(report["confusion_matrix"], dtype=float)
    shares = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, label="share of the true class")

    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=60, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if shares[i, j] >= 0.01:
                ax.text(j, i, f"{shares[i, j] * 100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if shares[i, j] > 0.5 else "black")
    fig.suptitle(f"{report['model']} on the {report['split']} split, row-normalised to percent")
    save(fig, path)


def describe(summary: RunSummary, history: list[EpochRecord], report: EvaluationReport, environment: dict,
             variant: str = "") -> dict:
    """One table row, unrounded so `summarize` does not take a deviation of rounded numbers."""
    return {
        "run_id": summary["run_id"],
        "model": summary["model"],
        "representation": summary["representation"],
        "variant": variant,
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
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["representation"], row.get("variant", "")), []).append(row)

    configurations = []
    for (model, representation, variant), members in groups.items():
        scores = [row["macro_f1"] for row in members]
        configurations.append(
            {
                "model": model,
                "representation": representation,
                "variant": variant,
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

    loaded = [load_run(run_dir) for run_dir in args.runs]
    summaries = [summary for _, summary, _ in loaded]
    labels = variants(summaries, [config for config, _, _ in loaded])

    per_model: dict[str, set[str]] = {}
    for summary, variant in zip(summaries, labels):
        per_model.setdefault(summary["model"], set()).add(configuration(summary, variant))

    histories: dict[str, list[list[EpochRecord]]] = {}
    seen: set[str] = set()
    matrices: dict[str, EvaluationReport] = {}
    rows = []
    for run_dir, (config, summary, history), variant in zip(args.runs, loaded, labels):
        report = report_path(config, summary["run_id"], split)
        if not report.is_file():
            parser.error(f"no {split} report for {summary['run_id']}; run python -m src.evaluate {summary['run_id']} --split {split}")

        row_label = label(summary, variant)
        if row_label in seen:
            parser.error(f"two runs given for {row_label}; a comparison takes one run per configuration and seed")
        seen.add(row_label)

        environment = shared_io.read_json(run_dir / "environment.json")
        rows.append(describe(summary, history, shared_io.read_json(report), environment, variant))
        key = configuration(summary, variant)
        histories.setdefault(key, []).append(history)
        name = summary["model"] if len(per_model[summary["model"]]) == 1 else re.sub(r"\W+", "_", key)
        matrices.setdefault(name, shared_io.read_json(report))

    configurations = summarize(rows)
    comparison = resolve_path(FIGURES) / (args.name or default_name(summaries))
    plot_curves(histories, comparison / "learning_curves")
    for name, report in matrices.items():
        plot_confusion(report, comparison / f"confusion_{name}_{split}")
    shared_io.write_json(
        comparison / f"metrics_{split}.json", {"split": split, "runs": rows, "configurations": configurations}
    )

    width = max((len(row["variant"]) for row in rows), default=0) + 2
    print(f"{'model':<12}{'repr':<11}{'variant':<{width}}{'seed':>5}{'params':>10}{'best':>6}"
          f"{'s/epoch':>9}{'accuracy':>10}{'macro-F1':>10}")
    for row in rows:
        print(
            f"{row['model']:<12}{row['representation']:<11}{row['variant']:<{width}}{row['seed']:>5}"
            f"{row['parameters']:>10,}{row['best_epoch']:>6}{row['seconds_per_epoch']:>9.1f}"
            f"{row['accuracy']:>10.4f}{row['macro_f1']:>10.4f}"
        )

    print(f"\n{'model':<12}{'repr':<11}{'variant':<{width}}{'seeds':>6}{'macro-F1 mean':>15}{'sd':>8}")
    for group in configurations:
        print(
            f"{group['model']:<12}{group['representation']:<11}{group['variant']:<{width}}"
            f"{len(group['seeds']):>6}{group['macro_f1_mean']:>15.4f}{group['macro_f1_sd']:>8.4f}"
        )

    devices = {row["hardware"] for row in rows}
    if len(devices) > 1:
        print(f"\nwarning: runs come from different machines, so the times are not comparable: {devices}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
