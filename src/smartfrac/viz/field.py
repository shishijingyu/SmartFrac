"""连续介质云图（FR-4.4/4.5）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_field_contour(coords: np.ndarray, values: np.ndarray,
                       title: str = "field", out: str | Path = "fig_field.png",
                       dpi: int = 300) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    tri = ax.tricontourf(coords[:, 0], coords[:, 1], values, levels=20, cmap="viridis")
    ax.tricontour(coords[:, 0], coords[:, 1], values, levels=10, colors="k",
                  linewidths=0.3)
    plt.colorbar(tri, ax=ax, label=title)
    ax.set_aspect("equal"); ax.set_title(title)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out)
