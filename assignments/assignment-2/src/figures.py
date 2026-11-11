"""Plots and tables derived from finished runs.

    python -m src.figures <run_id> [...] [--name baselines] [--split val]

Writes one directory per comparison under `results/figures/`. Metrics come from the evaluation
reports, not from `history.json`. The averaging, styling, confusion plot and decision rule live
in `shared.figures`.
"""

from __future__ import annotations

import argparse
import re
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shared import io as shared_io
from shared.figures import (
    COLOURS, default_name, load_run as _load_run, plot_confusion,
    plot_curves as _plot_curves, save, summarize, variants,
)

from .interfaces import CLASS_NAMES, EVALUATION_SPLITS, EpochRecord, EvaluationReport, RunSummary, Split
from .utils import report_path, resolve_path, run_dir as resolve_run

FIGURES = "results/figures"

PANELS = (("loss", "training objective"), ("miou", "mean IoU"))
"""Learning-curve panels. The loss axis is named for the objective, since the ablation uses Dice."""

VARIANT_KEYS = ("train.loss",)
"""Settings outside `model.args` that distinguish two runs of the same model."""


def load_run(directory: Path) -> tuple[dict, RunSummary, list[EpochRecord]]:
    """Resolved configuration, summary, and per-epoch history of one finished run."""
    return _load_run(directory)


ARM_NAMES = {
    "freeze_backbone=True": "frozen backbone",
    "loss=dice": "Dice loss",
}
"""Short names for the settings an experiment varies, so a legend stays readable."""


def configuration(summary: RunSummary, variant: str = "") -> str:
    """Curves are grouped by configuration, not by run; seeds of one configuration average.

    The name is the architecture, its backbone, and only those varied settings that actually
    distinguish this arm: `variants` lists every key that differs anywhere in the comparison,
    which at five arms is too long to label a legend with.
    """
    name = {"unet": "U-Net", "deeplabv3": "DeepLabV3"}.get(summary["model"], summary["model"])
    backbone = {"mobilenet_v3_large": "MobileNetV3", "resnet50": "ResNet50"}.get(
        summary["backbone"], summary["backbone"]
    )
    if backbone:
        name = f"{name}-{backbone}"
    marks = [ARM_NAMES[setting] for setting in variant.split(",") if setting in ARM_NAMES]
    return f"{name}, {', '.join(marks)}" if marks else name


def label(summary: RunSummary, variant: str = "") -> str:
    """Table row label, unique across seeds, backbones and model variants."""
    return f"{configuration(summary, variant)} seed{summary['seed']}"


def plot_curves(runs: dict[str, list[list[EpochRecord]]], path: Path) -> None:
    """Training and validation loss and mIoU for every run, on shared axes."""
    _plot_curves(runs, path, PANELS)


def plot_per_class(reports: dict[str, EvaluationReport], path: Path) -> None:
    """Per-class IoU per configuration, ordered by the hardest class.

    This is where mIoU and pixel accuracy disagree: a model can score well overall and still
    leave a class at exactly zero, which a single mean hides.
    """
    order = sorted(CLASS_NAMES, key=lambda name: max(r["per_class_iou"].get(name) or 0.0 for r in reports.values()))
    positions = np.arange(len(order))
    width = 0.8 / max(len(reports), 1)

    fig, ax = plt.subplots(figsize=(max(9.0, 0.6 * len(order)), 4.6))
    for index, ((name, report), colour) in enumerate(zip(reports.items(), cycle(COLOURS))):
        values = [report["per_class_iou"].get(klass) for klass in order]
        offsets = positions + index * width
        ax.bar(offsets, [value or 0.0 for value in values], width=width, color=colour, label=name)

        # An absent class has no IoU, which is why mean IoU is a nanmean. It is marked rather
        # than left blank, so it cannot be read as the zero that a missed class scores.
        missing = [offset for offset, value in zip(offsets, values) if value is None]
        ax.plot(missing, [0.02] * len(missing), linestyle="none", marker="x", markersize=4, color=colour)

    ax.set_xticks(positions + 0.4 - width / 2, order, rotation=60, ha="right")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Per-class IoU, classes ordered from hardest to easiest")
    save(fig, path)


def describe(summary: RunSummary, history: list[EpochRecord], report: EvaluationReport, environment: dict,
             variant: str = "") -> dict:
    """One table row, unrounded so `summarize` does not take a deviation of rounded numbers."""
    return {
        "run_id": summary["run_id"],
        "model": summary["model"],
        "backbone": summary["backbone"],
        "variant": variant,
        "seed": summary["seed"],
        "parameters": report["parameters"],
        "best_epoch": summary["best_epoch"],
        "epochs_trained": summary["epochs_trained"],
        "training_seconds": summary["training_seconds"],
        "seconds_per_epoch": sum(record["seconds"] for record in history) / len(history),
        "miou": report["miou"],
        "dice": report["dice"],
        "pixel_accuracy": report["pixel_accuracy"],
        "inference_ms_per_image": report["inference_ms_per_image"],
        "peak_memory_bytes": report.get("peak_memory_bytes"),
        "hardware": summary["hardware"],
        "torch": environment.get("torch"),
        "torchvision": environment.get("torchvision"),
        "python": environment.get("python"),
        "torch_threads": environment.get("torch_threads"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export learning curves and the metrics table for finished runs.")
    parser.add_argument("runs", nargs="+", type=resolve_run, help="run ids, or paths to the run directories")
    parser.add_argument("--name", default=None, help="comparison directory; defaults to the models compared")
    parser.add_argument("--split", choices=EVALUATION_SPLITS, default="val", help="which reports to tabulate")
    args = parser.parse_args(argv)
    split: Split = args.split

    loaded = [load_run(directory) for directory in args.runs]
    summaries = [summary for _, summary, _ in loaded]
    labels = variants(summaries, [config for config, _, _ in loaded], extra=VARIANT_KEYS)

    histories: dict[str, list[list[EpochRecord]]] = {}
    reports: dict[str, EvaluationReport] = {}
    rows: list[dict] = []
    seen: set[str] = set()

    for directory, (config, summary, history), variant in zip(args.runs, loaded, labels):
        report = report_path(config, summary["run_id"], split)
        if not report.is_file():
            parser.error(
                f"no {split} report for {summary['run_id']}; "
                f"run python -m src.evaluate {summary['run_id']} --split {split}"
            )

        row_label = label(summary, variant)
        if row_label in seen:
            parser.error(f"two runs given for {row_label}; a comparison takes one run per configuration and seed")
        seen.add(row_label)

        environment = shared_io.read_json(directory / "environment.json")
        rows.append(describe(summary, history, shared_io.read_json(report), environment, variant))

        key = configuration(summary, variant)
        histories.setdefault(key, []).append(history)
        reports.setdefault(key, shared_io.read_json(report))

    configurations = summarize(
        rows,
        metrics=("miou", "dice", "pixel_accuracy", "seconds_per_epoch"),
        keys=("model", "backbone", "variant"),
        carry=("parameters",),
    )

    comparison = resolve_path(FIGURES) / (args.name or default_name(summaries))
    plot_curves(histories, comparison / "learning_curves")
    plot_per_class(reports, comparison / f"per_class_iou_{split}")
    for name, report in reports.items():
        plot_confusion(
            report["confusion_matrix"], CLASS_NAMES,
            comparison / f"confusion_{re.sub(r'\W+', '_', name).strip('_').lower()}_{split}",
            f"{name} on the {report['split']} split, row-normalised to percent",
        )

    shared_io.write_json(
        comparison / f"metrics_{split}.json",
        {"split": split, "runs": rows, "configurations": configurations, "cost": _cost(rows)},
    )

    _print_runs(rows)
    _print_configurations(rows, configurations)

    devices = {row["hardware"] for row in rows}
    if len(devices) > 1:
        print(f"\nwarning: runs come from different machines, so the times are not comparable: {devices}")
    return 0


def _cost(rows: list[dict]) -> dict:
    """The compute-cost table of #73, with the hardware and versions that make it meaningful."""
    return {
        "hardware": sorted({row["hardware"] for row in rows}),
        "software": sorted({f"python {row['python']}, torch {row['torch']}, "
                            f"torchvision {row['torchvision']}" for row in rows}),
        "configurations": [
            {key: row[key] for key in (
                "model", "backbone", "variant", "seed", "parameters", "training_seconds",
                "seconds_per_epoch", "inference_ms_per_image", "peak_memory_bytes",
            )}
            for row in rows
        ],
    }


def _gigabytes(value: int | None) -> str:
    return "" if value is None else f"{value / 1024 ** 3:.2f}"


def _print_runs(rows: list[dict]) -> None:
    width = max((len(row["variant"]) for row in rows), default=0) + 2
    sizes = sorted({size for row in rows for size in row["inference_ms_per_image"]}, key=int)

    print(f"{'model':<11}{'backbone':<20}{'variant':<{width}}{'seed':>5}{'params':>12}{'best':>6}"
          f"{'s/epoch':>9}{'mIoU':>8}{'Dice':>8}{'pixacc':>8}{'peak GB':>9}"
          + "".join(f"{'ms@' + size:>9}" for size in sizes))

    for row in rows:
        latency = "".join(f"{row['inference_ms_per_image'].get(size, float('nan')):>9.1f}" for size in sizes)
        print(f"{row['model']:<11}{row['backbone']:<20}{row['variant']:<{width}}{row['seed']:>5}"
              f"{row['parameters']:>12,}{row['best_epoch']:>6}{row['seconds_per_epoch']:>9.1f}"
              f"{row['miou']:>8.4f}{row['dice']:>8.4f}{row['pixel_accuracy']:>8.4f}"
              f"{_gigabytes(row['peak_memory_bytes']):>9}" + latency)


def _print_configurations(rows: list[dict], configurations: list[dict]) -> None:
    width = max((len(row["variant"]) for row in rows), default=0) + 2
    print(f"\n{'model':<11}{'backbone':<20}{'variant':<{width}}{'seeds':>6}"
          f"{'mIoU mean':>11}{'sd':>8}{'Dice':>8}{'pixacc':>8}")
    for group in configurations:
        print(f"{group['model']:<11}{group['backbone']:<20}{group['variant']:<{width}}"
              f"{len(group['seeds']):>6}{group['miou_mean']:>11.4f}{group['miou_sd']:>8.4f}"
              f"{group['dice_mean']:>8.4f}{group['pixel_accuracy_mean']:>8.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
