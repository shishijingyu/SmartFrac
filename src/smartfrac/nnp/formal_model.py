"""正式版 NNP：双头模型（每原子能量贡献 + 每原子力）。

输入: (N_atoms, 147) SOAP 描述符
能量头: 147 -> 128 -> 64 -> 1  (每原子能量贡献)
力头:   147 -> 128 -> 64 -> 3  (每原子 xyz 力)
总能量 = sum(能量头输出)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NNPNet(nn.Module):
    def __init__(self, in_dim: int = 147, hidden: int = 128):
        super().__init__()
        self.energy_head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
        self.force_head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, 147) -> (per_atom_energy, per_atom_force)."""
        e_per_atom = self.energy_head(x).squeeze(-1)  # (N,)
        f_per_atom = self.force_head(x)               # (N, 3)
        return e_per_atom, f_per_atom

    def total_energy(self, e_per_atom: torch.Tensor) -> torch.Tensor:
        return e_per_atom.sum()
