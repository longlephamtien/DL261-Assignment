"""Plots and tables derived from finished runs.

    python -m src.figures <run_id> [...] [--name baselines] [--split val]
    python -m src.figures --pipeline

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from shared import io as shared_io
from shared.figures import (
    COLOURS, default_name, load_run as _load_run, plot_confusion,
    plot_curves as _plot_curves, save, separated, summarize, variants,
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


def per_class_means(reports: list[EvaluationReport]) -> dict[str, float | None]:
    """Mean IoU of each class over the seeds of one configuration, None where it is absent."""
    means: dict[str, float | None] = {}
    for name in CLASS_NAMES:
        scores = [r["per_class_iou"][name] for r in reports if r["per_class_iou"].get(name) is not None]
        means[name] = sum(scores) / len(scores) if scores else None
    return means


def plot_per_class(grouped: dict[str, list[EvaluationReport]], path: Path) -> None:
    """Per-class IoU per configuration, averaged over seeds, ordered by the hardest class.

    This is where mIoU and pixel accuracy disagree: a model can score well overall and still
    leave a class at exactly zero, which a single mean hides.
    """
    averaged = {name: per_class_means(reports) for name, reports in grouped.items()}
    order = sorted(CLASS_NAMES, key=lambda name: max(m.get(name) or 0.0 for m in averaged.values()))
    positions = np.arange(len(order))
    width = 0.8 / max(len(averaged), 1)

    fig, ax = plt.subplots(figsize=(max(9.0, 0.6 * len(order)), 4.6))
    for index, ((name, means), colour) in enumerate(zip(averaged.items(), cycle(COLOURS))):
        values = [means[klass] for klass in order]
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
    fig.suptitle("Per-class IoU, averaged over seeds, hardest class first")
    save(fig, path)


def describe(summary: RunSummary, history: list[EpochRecord], report: EvaluationReport, environment: dict,
             variant: str = "") -> dict:
    """One table row, unrounded so `summarize` does not take a deviation of rounded numbers."""
    return {
        "run_id": summary["run_id"],
        "name": configuration(summary, variant),
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


BLUE, DARK, RUST, GREEN, GREY = "#1488D8", "#030391", "#C1440E", "#2E7D32", "#55606B"

# (x, y, width, height, title, detail, colour)
BOXES = [
    (0.4, 6.1, 2.5, 1.0, "Raw data", "Cityscapes fine\n5,000 frames, 1024x2048", GREY),
    (3.5, 6.1, 2.5, 1.0, "Preprocessing", "resize 512x1024, label\nremap to 19 + ignore", GREY),
    (6.6, 6.1, 2.5, 1.0, "Data loader", "random 512 crop, flip,\nImageNet normalization", GREY),
    (9.7, 6.1, 2.5, 1.0, "Model", "U-Net  |  DeepLabV3\n3x512x512 -> 19x512x512", BLUE),
    (9.7, 4.3, 2.5, 1.0, "Loss", "cross-entropy or Dice\nignore_index=255", RUST),
    (6.6, 4.3, 2.5, 1.0, "Optimization", "AdamW 3e-4, cosine\nover 40 epochs", RUST),
    (3.5, 4.3, 2.5, 1.0, "Prediction", "argmax over classes\nper pixel", BLUE),
    (0.4, 4.3, 2.5, 1.0, "Post-processing", "none: logits are already\nat input resolution", GREY),
    (0.4, 2.3, 2.5, 1.0, "Evaluation", "mIoU, Dice, per-class IoU\nlatency, peak memory", GREEN),
    (3.5, 2.3, 2.5, 1.0, "Checkpoint", "highest validation mIoU,\npatience 8", GREEN),
    (6.6, 2.3, 5.6, 1.0, "Reporting", "3 seeds per arm, mean +/- 1 sd, decision rule;\ntest split read once after selection", DARK),
]

# (from, to, label); indices into BOXES
ARROWS = [(0, 1, ""), (1, 2, ""), (2, 3, ""), (3, 4, ""), (4, 5, ""),
          (5, 6, ""), (6, 7, ""), (7, 8, ""), (8, 9, ""), (9, 10, "")]


def _box(ax, x, y, w, h, title, detail, colour):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                linewidth=1.4, edgecolor=colour, facecolor=colour + "14"))
    ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color=colour)
    ax.text(x + w / 2, y + 0.28, detail, ha="center", va="center", fontsize=7.2, color="#333333")


def _edge(index):
    x, y, w, h = BOXES[index][:4]
    return {"left": (x, y + h / 2), "right": (x + w, y + h / 2),
            "top": (x + w / 2, y + h), "bottom": (x + w / 2, y)}


def _arrow(ax, start, end):
    """Route along the row, or down between rows, so no arrow crosses a box."""
    a, b = _edge(start), _edge(end)
    ay, by = BOXES[start][1], BOXES[end][1]
    if ay == by:
        tail, head = (a["right"], b["left"]) if BOXES[start][0] < BOXES[end][0] else (a["left"], b["right"])
        style = "arc3,rad=0"
    else:
        tail, head = a["bottom"], b["top"]
        style = "arc3,rad=0" if abs(tail[0] - head[0]) < 0.05 else "angle,angleA=-90,angleB=90,rad=8"
    ax.add_patch(FancyArrowPatch(tail, head, connectionstyle=style, arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.2, color="#55606B"))


def plot_pipeline(path: Path) -> None:
    """The pipeline diagram of the report, which depends on no run."""
    fig, ax = plt.subplots(figsize=(12.6, 4.9))
    for box in BOXES:
        _box(ax, *box)
    for start, end, _ in ARROWS:
        _arrow(ax, start, end)

    ax.text(12.4, 6.6, "train", fontsize=8, style="italic", color=BLUE, ha="left", va="center")
    ax.text(12.4, 4.8, "per batch", fontsize=8, style="italic", color=RUST, ha="left", va="center")
    ax.text(0.4, 1.95, "per epoch, on validation", fontsize=8, style="italic", color=GREEN,
            ha="left", va="top")

    ax.set_xlim(0, 13.4)
    ax.set_ylim(1.75, 7.35)
    ax.axis("off")
    return fig

# Error groups, applied in this order so every error pixel falls in exactly one. The split is by
# what the confusion is between, not by which class is at fault, because almost every group is
# dominated by boundary pixels that belong to both.
THIN = frozenset({"pole", "traffic light", "traffic sign"})
GROUND = frozenset({"road", "sidewalk", "terrain"})
UPRIGHT = frozenset({"building", "wall", "fence", "vegetation"})
MOVERS = frozenset({"car", "truck", "bus", "train", "motorcycle", "bicycle", "rider", "person"})
ERROR_GROUPS = ("thin structures", "ground surfaces", "upright surfaces", "traffic participants",
                "participant vs. background", "other")


def error_group(truth: str, prediction: str) -> str:
    """The group one confusion belongs to."""
    if truth in THIN or prediction in THIN:
        return "thin structures"
    if truth in GROUND and prediction in GROUND:
        return "ground surfaces"
    if truth in UPRIGHT and prediction in UPRIGHT:
        return "upright surfaces"
    if truth in MOVERS and prediction in MOVERS:
        return "traffic participants"
    if truth in MOVERS or prediction in MOVERS:
        return "participant vs. background"
    return "other"


def _error_groups(reports: list[EvaluationReport]) -> dict:
    """Error pixels per group, as a share of this configuration's own errors."""
    matrix = np.mean([np.array(r["confusion_matrix"], dtype=float) for r in reports], axis=0)
    off_diagonal = matrix - np.diag(np.diag(matrix))
    errors = off_diagonal.sum()
    shares = dict.fromkeys(ERROR_GROUPS, 0.0)
    for truth in range(len(CLASS_NAMES)):
        for prediction in range(len(CLASS_NAMES)):
            if truth != prediction:
                shares[error_group(CLASS_NAMES[truth], CLASS_NAMES[prediction])] += off_diagonal[truth, prediction]
    return {
        "error_rate": float(errors / matrix.sum()) if matrix.sum() else 0.0,
        "error_pixels": float(errors),
        "groups": {name: float(value / errors) if errors else 0.0 for name, value in shares.items()},
    }


def _abbrev(name: str, limit: int) -> str:
    """Shorten a name for a printed column, eliding the middle.

    Configuration names share a long prefix and differ at the end, so cutting the tail would
    make several columns identical.
    """
    if len(name) <= limit:
        return name
    head = (limit - 1) // 2
    return name[:head] + "\u2026" + name[len(name) - (limit - 1 - head):]


def _print_verdicts(configurations: list[dict]) -> None:
    """Every pairwise comparison under the decision rule, on the metric that selected checkpoints."""
    print(f"\n{'comparison':<56}verdict")
    for index, first in enumerate(configurations):
        for second in configurations[index + 1:]:
            pair = _abbrev(f"{first['name']} vs. {second['name']}", 54)
            if len(first["seeds"]) < 2 or len(second["seeds"]) < 2:
                verdict = "indicative only, one seed"
            elif separated(first, second, "miou"):
                high, low = sorted((first, second), key=lambda c: -c["miou_mean"])
                verdict = f"{high['name']} better by {high['miou_mean'] - low['miou_mean']:.4f}"
            else:
                verdict = f"indistinguishable ({abs(first['miou_mean'] - second['miou_mean']):.4f})"
            print(f"{pair:<56}{verdict}")


def _print_per_class(reports: dict[str, list[EvaluationReport]]) -> None:
    """Per-class IoU per configuration, averaged over seeds, hardest class first."""
    averaged = {name: per_class_means(group) for name, group in reports.items()}
    order = sorted(CLASS_NAMES, key=lambda name: max(m.get(name) or 0.0 for m in averaged.values()))
    width = min(max(len(name) for name in averaged), 20) + 2
    print(f"\n{'class':<15}" + "".join(f"{_abbrev(name, width - 2):>{width}}" for name in averaged))
    for klass in order:
        row = f"{klass:<15}"
        for means in averaged.values():
            row += f"{means[klass]:>{width}.4f}" if means[klass] is not None else f"{'--':>{width}}"
        print(row)


def _print_errors(reports: dict[str, list[EvaluationReport]]) -> None:
    """The error taxonomy: where each configuration's mistakes fall, as a share of its own."""
    grouped = {name: _error_groups(group) for name, group in reports.items()}
    width = min(max(len(name) for name in grouped), 20) + 2
    print(f"\n{'error group':<28}" + "".join(f"{_abbrev(name, width - 2):>{width}}" for name in grouped))
    for group in ERROR_GROUPS:
        print(f"{group:<28}" + "".join(f"{g['groups'][group]:>{width}.1%}" for g in grouped.values()))
    print(f"{'error rate (all pixels)':<28}" + "".join(f"{g['error_rate']:>{width}.1%}" for g in grouped.values()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export learning curves and the metrics table for finished runs.")
    parser.add_argument("runs", nargs="*", type=resolve_run, help="run ids, or paths to the run directories")
    parser.add_argument("--pipeline", action="store_true", help="write the pipeline diagram and stop")
    parser.add_argument("--name", default=None, help="comparison directory; defaults to the models compared")
    parser.add_argument("--split", choices=EVALUATION_SPLITS, default="val", help="which reports to tabulate")
    args = parser.parse_args(argv)
    split: Split = args.split

    if args.pipeline:
        plot_pipeline(resolve_path(FIGURES) / "pipeline")
        return 0
    if not args.runs:
        parser.error("give at least one run, or --pipeline")

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
        reports.setdefault(key, []).append(shared_io.read_json(report))

    configurations = summarize(
        rows,
        metrics=("miou", "dice", "pixel_accuracy", "seconds_per_epoch"),
        keys=("model", "backbone", "variant"),
        carry=("parameters", "name"),
    )

    comparison = resolve_path(FIGURES) / (args.name or default_name(summaries))
    plot_curves(histories, comparison / "learning_curves")
    plot_per_class(reports, comparison / f"per_class_iou_{split}")
    for name, group in reports.items():
        plot_confusion(
            np.mean([np.array(r["confusion_matrix"], dtype=float) for r in group], axis=0).tolist(),
            CLASS_NAMES,
            comparison / f"confusion_{re.sub(r'\W+', '_', name).strip('_').lower()}_{split}",
            f"{name} on the {split} split, row-normalised to percent",
        )

    shared_io.write_json(
        comparison / f"metrics_{split}.json",
        {"split": split, "runs": rows, "configurations": configurations,
         "cost": _cost(rows), "errors": {name: _error_groups(group) for name, group in reports.items()}},
    )

    _print_runs(rows)
    _print_configurations(rows, configurations)
    _print_verdicts(configurations)
    _print_per_class(reports)
    _print_errors(reports)

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
