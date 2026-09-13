"""曲线类图（FR-4.3/4.6/4.7）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(history: dict, out: str | Path = "fig_loss.png",
                    dpi: int = 300) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in history.items():
        ax.plot(v, label=k)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_yscale("log")
    ax.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out)


def parity_plot(pred: np.ndarray, ref: np.ndarray,
                out: str | Path = "fig_parity.png",
                dpi: int = 300) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ref, pred, s=10, alpha=0.6)
    lo = min(ref.min(), pred.min()); hi = max(ref.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y=x")
    ax.set_xlabel("reference"); ax.set_ylabel("predicted"); ax.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out)


def plot_profile(x: np.ndarray, y_ref: np.ndarray, y_pred: np.ndarray,
                 label: str = "displacement", out: str | Path = "fig_profile.png",
                 dpi: int = 300) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_ref, "k-", label="reference")
    ax.plot(x, y_pred, "r--", label="PINN")
    ax.set_xlabel("x"); ax.set_ylabel(label); ax.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out)
