"""Element-specific 原子能量网络（FR-2.1/2.2）。

E_total = Σ_i E_i(descr_i, species_i)
F_i = -∂E_total / ∂r_i
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from smartfrac.core.base_model import BaseModel


def _mlp(in_dim: int, widths: list[int], act: str) -> nn.Sequential:
    acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}
    layers: list[nn.Module] = []
    prev = in_dim
    for w in widths:
        layers += [nn.Linear(prev, w), acts[act]()]
        prev = w
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers)


class ElementEnergyNet(BaseModel):
    """每种元素一个小 MLP，输入描述符输出原子能量贡献。"""

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__(cfg)
        d = self.cfg.get("descriptor_dim", 300)
        widths = self.cfg.get("hidden_dims", [128, 128, 64])
        act = self.cfg.get("activation", "tanh")
        species = self.cfg.get("species", [26])  # 默认 Fe
        self.species = species
        self.heads = nn.ModuleDict({
            str(z): _mlp(d, widths, act) for z in species
        })

    def forward(self, desc: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        """desc: (N_atoms, D); species: (N_atoms,) int -> 每原子能量 (N_atoms,)。"""
        out = torch.zeros(len(desc), device=desc.device)
        for z, head in self.heads.items():
            mask = species == int(z)
            if mask.any():
                out[mask] = head(desc[mask]).squeeze(-1)
        return out

    def energy_total(self, per_atom_energy: torch.Tensor) -> torch.Tensor:
        return per_atom_energy.sum()
