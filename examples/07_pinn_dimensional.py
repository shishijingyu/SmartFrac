"""TC-05: PINN 无量纲化 + Westergaard 解析解对比。

单位制: mm, GPa, (应力/模量) -> 位移无量纲
坐标范围: -10mm ~ +10mm (裂纹长 a=5mm)
和 Westergaard 解析解对比 L2 误差, 目标 < 2%.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from smartfrac.constants import (
    CRACK_HALF_LENGTH, E, NU, SIGMA_INF,
)

OUT = ROOT / "results" / "pinn_dimensional"
OUT.mkdir(parents=True, exist_ok=True)

# 物理参数 (mm/GPa 单位)
A = CRACK_HALF_LENGTH * 1000  # mm
E_GPa = E / 1e9                # GPa
SIG_inf = SIGMA_INF / 1e9      # GPa
nu = NU
DOMAIN = 20.0  # mm


# ---------- Westergaard I 型裂纹解析解 ----------
def westergaard_u(x, y):
    """位移场 (mm), I 型裂纹. x,y 单位 mm."""
    z = x + 1j * y
    za = z + A
    sqrt_za = np.sqrt(za)
    # Westergaard 应力函数 ZI = SIG_inf / sqrt(1 - (a/z)^2)
    ZI = SIG_inf / np.sqrt(1 - (A / z) ** 2)
    # 位移 (平面应力)
    # ux = (1/E) * [Re(Zbar) - nu*Im(Zbar)] ... 简化
    ux = (1 + nu) / E_GPa * (np.cos(np.angle(sqrt_za)) * np.abs(sqrt_za) - A)
    uy = (1 + nu) / E_GPa * np.sin(np.angle(sqrt_za)) * np.abs(sqrt_za)
    return ux * A, uy * A  # mm


# ---------- PINN ----------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    model = PINN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 配点: 在域内均匀采样, 避开裂纹线
    n_col = 5000
    x_col = (torch.rand(n_col) * 2 - 1) * DOMAIN
    y_col = (torch.rand(n_col) * 2 - 1) * DOMAIN
    mask = ~((torch.abs(y_col) < 0.1) & (torch.abs(x_col) < A))
    x_col, y_col = x_col[mask], y_col[mask]
    x_col.requires_grad_(True)
    y_col.requires_grad_(True)

    # 边界点: 远场应力 (简化: ux=0 on left, uy=0.2mm on top)
    # 这里直接用解析解做边界监督
    n_bc = 500
    bc_x = torch.cat([
        (torch.rand(n_bc // 4) * 2 - 1) * DOMAIN,  # bottom
        torch.ones(n_bc // 4) * DOMAIN,             # right
        (torch.rand(n_bc // 4) * 2 - 1) * DOMAIN,  # top
        -torch.ones(n_bc // 4) * DOMAIN,            # left
    ])
    bc_y = torch.cat([
        -torch.ones(n_bc // 4) * DOMAIN,
        (torch.rand(n_bc // 4) * 2 - 1) * DOMAIN,
        torch.ones(n_bc // 4) * DOMAIN,
        (torch.rand(n_bc // 4) * 2 - 1) * DOMAIN,
    ])

    # 解析解标签
    with torch.no_grad():
        ux_bc, uy_bc = westergaard_u(bc_x.numpy(), bc_y.numpy())
        bc_u = torch.tensor(np.column_stack([ux_bc, uy_bc]), dtype=torch.float32)

    history = {"loss": []}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000, eta_min=1e-5)
    for ep in range(1, 5001):
        model.train()
        opt.zero_grad()

        # 边界损失
        pred_bc = model(torch.column_stack([bc_x, bc_y]))
        loss_bc = ((pred_bc - bc_u) ** 2).mean()

        # PDE 损失 (简化: 线弹性平衡方程 div σ = 0)
        pred = model(torch.column_stack([x_col, y_col]))
        ux, uy = pred[:, 0], pred[:, 1]
        # 应变
        e_xx = torch.autograd.grad(ux.sum(), x_col, create_graph=True)[0]
        e_yy = torch.autograd.grad(uy.sum(), y_col, create_graph=True)[0]
        e_xy = 0.5 * (torch.autograd.grad(ux.sum(), y_col, create_graph=True)[0]
                      + torch.autograd.grad(uy.sum(), x_col, create_graph=True)[0])
        # 应力 (平面应力)
        lam = E_GPa * nu / (1 - nu ** 2)
        mu = E_GPa / (2 * (1 + nu))
        s_xx = lam * (e_xx + e_yy) + 2 * mu * e_xx
        s_yy = lam * (e_xx + e_yy) + 2 * mu * e_yy
        s_xy = 2 * mu * e_xy
        # 平衡
        f1 = torch.autograd.grad(s_xx.sum(), x_col, create_graph=True)[0] \
             + torch.autograd.grad(s_xy.sum(), y_col, create_graph=True)[0]
        f2 = torch.autograd.grad(s_xy.sum(), x_col, create_graph=True)[0] \
             + torch.autograd.grad(s_yy.sum(), y_col, create_graph=True)[0]
        loss_pde = (f1 ** 2).mean() + (f2 ** 2).mean()

        loss = loss_bc + 0.5 * loss_pde
        loss.backward()
        opt.step()
        sched.step()
        history["loss"].append(float(loss.detach()))

        if ep % 500 == 0:
            print(f"ep {ep}: loss={loss.item():.4e}  bc={loss_bc.item():.4e}  pde={loss_pde.item():.4e}")

    # 评估: 和 Westergaard 对比 L2
    model.eval()
    nx, ny = 40, 40
    xs = np.linspace(-DOMAIN, DOMAIN, nx)
    ys = np.linspace(-DOMAIN, DOMAIN, ny)
    X, Y = np.meshgrid(xs, ys)
    with torch.no_grad():
        pred = model(torch.tensor(np.column_stack([X.ravel(), Y.ravel()]),
                                  dtype=torch.float32)).numpy()
    ux_pred = pred[:, 0].reshape(nx, ny)
    uy_pred = pred[:, 1].reshape(nx, ny)
    ux_ref, uy_ref = westergaard_u(X, Y)

    l2 = np.sqrt(((ux_pred - ux_ref) ** 2 + (uy_pred - uy_ref) ** 2).mean())
    scale = np.sqrt((ux_ref ** 2 + uy_ref ** 2).mean())
    rel_l2 = l2 / scale
    print(f"\n=== PINN vs Westergaard ===")
    print(f"绝对 L2 = {l2:.4e} mm")
    print(f"相对 L2 = {rel_l2*100:.2f}%  (TC-05 目标 < 2%)")

    # 画图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    vmax = np.abs(uy_ref).max()
    im = axes[0, 0].contourf(X, Y, ux_ref, 20, cmap="RdBu_r")
    axes[0, 0].set_title("Westergaard ux (mm)"); plt.colorbar(im, ax=axes[0, 0])
    im = axes[0, 1].contourf(X, Y, ux_pred, 20, cmap="RdBu_r")
    axes[0, 1].set_title("PINN ux (mm)"); plt.colorbar(im, ax=axes[0, 1])
    im = axes[1, 0].contourf(X, Y, uy_ref, 20, cmap="RdBu_r")
    axes[1, 0].set_title("Westergaard uy (mm)"); plt.colorbar(im, ax=axes[1, 0])
    im = axes[1, 1].contourf(X, Y, uy_pred, 20, cmap="RdBu_r")
    axes[1, 1].set_title("PINN uy (mm)"); plt.colorbar(im, ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig(OUT / "field_comparison.png", dpi=100)
    print(f"图 -> {OUT / 'field_comparison.png'}")


if __name__ == "__main__":
    main()
