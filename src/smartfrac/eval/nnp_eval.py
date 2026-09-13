"""NNP 验证（接口 B）。"""
from __future__ import annotations

import numpy as np

from smartfrac.nnp.metrics import nnp_metrics


def evaluate_nnp(pred_energies: np.ndarray, true_energies: np.ndarray,
                pred_forces: np.ndarray, true_forces: np.ndarray,
                speedup: float | None = None) -> dict:
    return nnp_metrics(pred_energies, true_energies,
                       pred_forces, true_forces, speedup)
