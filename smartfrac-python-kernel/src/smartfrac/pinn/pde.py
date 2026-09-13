"""二维线弹性 PDE（FR-3.2）：autograd 推导应变/应力/残差。"""
from __future__ import annotations

import torch


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, torch.ones_like(y),
                               create_graph=True)[0]


def elastic_residual(coords: torch.Tensor, model, E: float, nu: float) -> torch.Tensor:
    """返回 (r_x, r_y) 平衡方程残差。平面应力。"""
    coords = coords.clone().requires_grad_(True)
    u = model(coords)
    ux, uy = u[:, 0:1], u[:, 1:2]

    # 一阶导
    dux = grad(ux, coords)
    duy = grad(uy, coords)
    uxx, uxy = dux[:, 0:1], dux[:, 1:2]
    uyx, uyy = duy[:, 0:1], duy[:, 1:2]

    # 应变
    exx = uxx
    eyy = uyy
    exy = 0.5 * (uxy + uyx)

    # 平面应力本构
    c = E / (1 - nu ** 2)
    sxx = c * (exx + nu * eyy)
    syy = c * (eyy + nu * exx)
    sxy = c * (1 - nu) * exy

    # 二阶导 -> 平衡方程
    dsxx_dx = grad(sxx, coords)[:, 0:1]
    dsxy_dy = grad(sxy, coords)[:, 1:2]
    dsxy_dx = grad(sxy, coords)[:, 0:1]
    dsyy_dy = grad(syy, coords)[:, 1:2]
    rx = dsxx_dx + dsxy_dy
    ry = dsxy_dx + dsyy_dy
    return rx, ry, sxx, syy, sxy
