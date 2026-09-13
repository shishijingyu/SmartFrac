"""PINN 验证（接口 C）。"""
from __future__ import annotations

import numpy as np


def evaluate_pinn(ux_pred: np.ndarray, uy_pred: np.ndarray,
                   ux_ref: np.ndarray, uy_ref: np.ndarray) -> dict:
    l2_x = float(np.linalg.norm(ux_pred - ux_ref) / (np.linalg.norm(ux_ref) + 1e-30))
    l2_y = float(np.linalg.norm(uy_pred - uy_ref) / (np.linalg.norm(uy_ref) + 1e-30))
    return {"L2_ux": l2_x, "L2_uy": l2_y,
            "L2_total": (l2_x + l2_y) / 2}
