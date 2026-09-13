"""边界条件（FR-3.3）。"""
from __future__ import annotations

import torch


def far_field_disp(coords: torch.Tensor, sigma_inf: float, E: float, nu: float,
                   a: float) -> torch.Tensor:
    """远场拉伸时的线性位移场（近似，作为 Dirichlet 参考）。

    薄板单向拉伸：ε_y = σ/E, u_y = ε_y * y, u_x = -nu*ε_y*x。
    裂纹长度 2a，远场边界在 |x|,|y| >> a 处。
    """
    ey = sigma_inf / E
    ex = -nu * ey
    ux = ex * coords[:, 0:1]
    uy = ey * coords[:, 1:2]
    return torch.cat([ux, uy], dim=1)
