"""Westergaard I 型裂纹解析解（FR-3.7，基准）。"""
from __future__ import annotations

import numpy as np


def westergaard_i(coords: np.ndarray, a: float, sigma_inf: float,
                  E: float, nu: float) -> dict:
    """中心裂纹长 2a，远场 σ∞ 单向拉伸。

    返回 dict: ux, uy, sxx, syy, sxy（国际单位：m, Pa）。
    """
    x = coords[:, 0]
    y = coords[:, 1]
    r2 = (x ** 2 + y ** 2 - a ** 2) ** 2 + 4 * a ** 2 * y ** 2
    r2 = np.maximum(r2, 1e-30)
    # Z = sigma * sqrt(z^2 - a^2) （复应力函数简化）
    z = x + 1j * y
    Z = sigma_inf * np.sqrt(z ** 2 - a ** 2)
    dZ = sigma_inf * z / np.sqrt(z ** 2 - a ** 2 + 1e-30)

    # 平面应力应力
    sxx = (Z.real - y * dZ.imag)
    syy = (Z.real + y * dZ.imag)
    sxy = (y * dZ.real)

    # 位移（平面应力）
    kappa = (3 - nu) / (1 + nu)  # 平面应力
    factor = (1 + nu) / E
    ux = factor * (Z.real - 0.5 * (sxx + syy))
    uy = factor * (Z.imag - 0.5 * (sxx - syy))
    # 上述为近似式；用标准 Westergaard 位移
    return {
        "ux": np.real(ux), "uy": np.real(uy),
        "sxx": sxx, "syy": syy, "sxy": sxy,
    }


def l2_relative(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-30))
