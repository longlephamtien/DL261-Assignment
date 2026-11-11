"""The pipeline diagram handbook section 3.2 prescribes, written once for the report and the site.

    python scripts/pipeline_figure.py

Writes results/figures/pipeline.{png,svg,pdf}.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from shared.figures import save

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


def build() -> plt.Figure:
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


if __name__ == "__main__":
    save(build(), ROOT / "results" / "figures" / "pipeline")
    print("wrote results/figures/pipeline.{png,svg,pdf}")
