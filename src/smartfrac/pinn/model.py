"""PINN MLP：(x,y)->(ux,uy)（FR-3.4）。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from smartfrac.core.base_model import BaseModel


class PINNMLP(BaseModel):
    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__(cfg)
        widths = self.cfg.get("hidden_dims", [64, 64, 64, 64])
        act = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}[
            self.cfg.get("activation", "tanh")]
        layers: list[nn.Module] = []
        prev = 2
        for w in widths:
            layers += [nn.Linear(prev, w), act()]
            prev = w
        layers += [nn.Linear(prev, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)
