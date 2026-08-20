"""Figure style and the panels the submission is built from.

Colours are chosen to stay legible in greyscale print, because a jury reads a PDF.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#1b1b1b"
GRID = "#d8d8d8"
ACCENT = "#0f6fc5"
WARM = "#c1461a"
MUTED = "#8a8f98"
GREEN = "#1f7a4d"
SAND = "#c9a227"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": False,
})


def save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax
