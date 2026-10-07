"""Figure export shared by every assignment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

COLOURS = ("#1488D8", "#030391", "#C1440E", "#2E7D32", "#6A1B9A")


def save(fig: plt.Figure, path: Path, close: bool = True) -> None:
    """PNG for the repository, SVG and PDF for the site and the report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    for suffix in (".svg", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=300)
    if close:
        plt.close(fig)
