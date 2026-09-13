"""NNP 指标（FR-2.10）。"""
from __future__ import annotations

import numpy as np


def nnp_metrics(pred_e: np.ndarray, true_e: np.ndarray,
                pred_f: np.ndarray, true_f: np.ndarray,
                speedup: float | None = None) -> dict:
    energy_mae = float(np.mean(np.abs(pred_e - true_e)) * 1000)  # meV/atom
    force_rmse = float(np.sqrt(np.mean((pred_f - true_f) ** 2)))  # eV/Å
    out = {"energy_mae_meV_per_atom": energy_mae,
           "force_rmse_eV_per_A": force_rmse}
    if speedup is not None:
        out["speedup"] = speedup
    return out
