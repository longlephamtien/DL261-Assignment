"""Figure export shared by every assignment."""

from __future__ import annotations

from itertools import cycle
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from . import io as shared_io

COLOURS = ("#1488D8", "#030391", "#C1440E", "#2E7D32", "#6A1B9A",
           "#B8860B", "#00838F", "#AD1457", "#455A64")


def save(fig: plt.Figure, path: Path, close: bool = True) -> None:
    """PNG for the repository, SVG and PDF for the site and the report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    for suffix in (".svg", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=300)
    if close:
        plt.close(fig)


def load_run(directory: Path) -> tuple[dict, dict, list[dict]]:
    """Resolved configuration, summary, and per-epoch history of one finished run."""
    directory = Path(directory)
    return (
        shared_io.read_yaml(directory / "config.yaml"),
        shared_io.read_json(directory / "summary.json"),
        shared_io.read_json(directory / "history.json"),
    )


def default_name(summaries: list[dict]) -> str:
    """Name a comparison after the models it covers."""
    return "_".join(sorted({summary["model"] for summary in summaries}))


def variants(summaries: list[dict], configs: list[dict], exclude: frozenset[str] = frozenset(),
             extra: tuple[str, ...] = ()) -> list[str]:
    """Settings that distinguish runs of the same model, such as the recurrent cell.

    Compared within a model name, so a comparison across architectures stays unlabelled.
    `exclude` drops a key that already labels the configuration; `extra` adds dotted paths
    outside `model.args`, such as the loss an ablation varies.
    """
    groups: dict[str, list[int]] = {}
    for index, summary in enumerate(summaries):
        groups.setdefault(summary["model"], []).append(index)

    labels = [""] * len(summaries)
    for indices in groups.values():
        args = [_variant_keys(configs[index], exclude, extra) for index in indices]
        keys = {key for arg in args for key in arg}
        varying = sorted(key for key in keys if len({repr(arg.get(key)) for arg in args}) > 1)
        for index, arg in zip(indices, args):
            labels[index] = ",".join(f"{key}={arg.get(key)}" for key in varying)
    return labels


def _variant_keys(config: dict, exclude: frozenset[str], extra: tuple[str, ...]) -> dict[str, Any]:
    """The settings of one run that a variant label may mention, keyed by their short name."""
    keys = {key: value for key, value in config.get("model", {}).get("args", {}).items() if key not in exclude}
    for dotted in extra:
        node: Any = config
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is not None:
            keys[dotted.rsplit(".", 1)[-1]] = node
    return keys


def band(histories: list[list[dict]], key: str) -> tuple[list[int], list[float], list[float]]:
    """Mean and standard deviation of one metric across seeds, epoch by epoch.

    Truncated to the shortest history: early stopping ends seeds at different epochs, and an
    average over a shrinking number of seeds would bend the curve.
    """
    length = min(len(history) for history in histories)
    epochs = [histories[0][index]["epoch"] for index in range(length)]
    values = [[history[index][key] for history in histories] for index in range(length)]
    means = [mean(value) for value in values]
    deviations = [stdev(value) if len(value) > 1 else 0.0 for value in values]
    return epochs, means, deviations


def plot_curves(runs: dict[str, list[list[dict]]], path: Path, panels: tuple[tuple[str, str], ...]) -> None:
    """Training and validation curves per configuration, averaged over its seeds.

    `panels` gives the metric key and axis label of each panel.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.4), squeeze=False)
    row = axes[0]

    for (name, histories), colour in zip(runs.items(), cycle(COLOURS)):
        seeds = len(histories)
        for ax, (key, _) in zip(row, panels):
            for split, style, width in (("val", "-", 1.8), ("train", "--", 1.0)):
                epochs, means, deviations = band(histories, f"{split}_{key}")
                ax.plot(epochs, means, color=colour, linestyle=style, linewidth=width,
                        label=name if split == "val" else None)
                if seeds > 1:
                    ax.fill_between(epochs,
                                    [m - d for m, d in zip(means, deviations)],
                                    [m + d for m, d in zip(means, deviations)],
                                    color=colour, alpha=0.15, linewidth=0)

    for ax, (_, ylabel) in zip(row, panels):
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    row[-1].legend(fontsize=8, loc="lower right")

    counts = {len(histories) for histories in runs.values()}
    shared_count = counts.pop() if len(counts) == 1 else 0
    spread = f", shaded band one standard deviation over {shared_count} seeds" if shared_count > 1 else ""
    fig.suptitle(f"Learning curves, solid validation and dashed training{spread}")
    save(fig, path)


def plot_confusion(matrix: list[list[int]], class_names: tuple[str, ...], path: Path, title: str) -> None:
    """Row-normalised confusion matrix: what each true class was predicted as.

    Normalised by row because raw counts hide which classes a model confuses. A row with no
    pixels stays blank rather than dividing by zero.
    """
    counts = np.array(matrix, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    size = max(7.2, 0.42 * len(class_names))
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, label="share of the true class")

    ax.set_xticks(range(len(class_names)), class_names, rotation=60, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")

    # Cells are annotated only where there is something to read; at 19 classes a label in
    # every cell is unreadable, and a near-zero share carries no information anyway.
    threshold = 0.01 if len(class_names) <= 12 else 0.02
    fontsize = 7 if len(class_names) <= 12 else 5.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if shares[i, j] >= threshold:
                ax.text(j, i, f"{shares[i, j] * 100:.0f}", ha="center", va="center", fontsize=fontsize,
                        color="white" if shares[i, j] > 0.5 else "black")
    fig.suptitle(title)
    save(fig, path)


def summarize(rows: list[dict], metrics: tuple[str, ...], keys: tuple[str, ...],
              carry: tuple[str, ...] = ()) -> list[dict]:
    """Mean and standard deviation of each metric per configuration, across its seeds.

    `keys` identify a configuration; `carry` are fields taken from its first member because
    they do not vary with the seed, such as the parameter count.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in keys), []).append(row)

    configurations: list[dict[str, Any]] = []
    for identity, members in groups.items():
        group: dict[str, Any] = {
            **dict(zip(keys, identity)),
            "seeds": sorted(row["seed"] for row in members),
            **{name: members[0][name] for name in carry},
        }
        for metric in metrics:
            scores = [row[metric] for row in members]
            group[f"{metric}_mean"] = mean(scores)
            group[f"{metric}_sd"] = stdev(scores) if len(scores) > 1 else 0.0
        configurations.append(group)
    return configurations


def separated(first: dict, second: dict, metric: str) -> bool:
    """Whether the decision rule separates two configurations on `metric`.

    One is superior only when its mean is higher and the two mean-plus-or-minus-one-standard-
    deviation intervals do not overlap; otherwise they are indistinguishable.
    """
    high, low = (first, second) if first[f"{metric}_mean"] >= second[f"{metric}_mean"] else (second, first)
    return high[f"{metric}_mean"] - high[f"{metric}_sd"] > low[f"{metric}_mean"] + low[f"{metric}_sd"]
