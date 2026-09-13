"""原子可视化（FR-4.1/4.2）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms


def plot_atoms_2d(atoms: Atoms, color_by: str = "element",
                  out: str | Path = "fig_atoms.png", dpi: int = 300) -> Path:
    pos = atoms.get_positions()
    fig, ax = plt.subplots(figsize=(6, 5))
    if color_by == "element":
        ax.scatter(pos[:, 0], pos[:, 1], c=atoms.get_atomic_numbers(),
                   cmap="tab20", s=20)
    else:
        ax.scatter(pos[:, 0], pos[:, 1], c=pos[:, 2], cmap="viridis", s=20)
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
    ax.set_aspect("equal")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out)


def plot_atom_scatter(coords: np.ndarray, values: np.ndarray,
                      title: str = "atomic field",
                      out: str | Path = "fig_atom_scatter.png",
                      dpi: int = 300) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap="viridis", s=15)
    plt.colorbar(sc, ax=ax, label=title)
    ax.set_aspect("equal")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out)
