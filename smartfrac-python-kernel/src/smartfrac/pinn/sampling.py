"""配点采样（FR-3.6）：均匀 + 裂尖 r² 加密。"""
from __future__ import annotations

import numpy as np
import torch


def sample_domain(n: int, Lx: float, Ly: float, crack_tip: tuple[float, float],
                  tip_fraction: float = 0.2, tip_radius: float = 0.5) -> torch.Tensor:
    """矩形域 [-Lx,Lx] x [-Ly,Ly]，裂尖附近 tip_fraction 比例按 r² 加密。"""
    n_tip = int(n * tip_fraction)
    n_uni = n - n_tip

    uni = np.column_stack([
        np.random.uniform(-Lx, Lx, n_uni),
        np.random.uniform(-Ly, Ly, n_uni),
    ])

    # 裂尖周围：r^2 分布
    r = tip_radius * np.sqrt(np.random.uniform(0, 1, n_tip))
    theta = np.random.uniform(0, 2 * np.pi, n_tip)
    tip = np.column_stack([
        crack_tip[0] + r * np.cos(theta),
        crack_tip[1] + r * np.sin(theta),
    ])
    pts = np.vstack([uni, tip])
    return torch.tensor(pts, dtype=torch.float32)


def sample_boundary(n: int, Lx: float, Ly: float) -> torch.Tensor:
    """四条边界均匀取点。"""
    pts = []
    for _ in range(n):
        edge = np.random.randint(4)
        if edge == 0:
            pts.append([-Lx, np.random.uniform(-Ly, Ly)])
        elif edge == 1:
            pts.append([Lx, np.random.uniform(-Ly, Ly)])
        elif edge == 2:
            pts.append([np.random.uniform(-Lx, Lx), -Ly])
        else:
            pts.append([np.random.uniform(-Lx, Lx), Ly])
    return torch.tensor(pts, dtype=torch.float32)
