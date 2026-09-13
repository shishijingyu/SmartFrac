"""PINN 训练器（FR-3.5）。"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..constants import E, NU, SIGMA_INF, CRACK_HALF_LENGTH
from .pde import elastic_residual
from .bc import far_field_disp

log = logging.getLogger(__name__)


class PINNTrainer:
    def __init__(self, model, cfg: dict, out_dir: str = "results"):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.w_pde = cfg.get("w_pde", 1.0)
        self.w_bc = cfg.get("w_bc", 10.0)
        self.epochs = cfg.get("epochs", 2000)
        self.lr = cfg.get("lr", 1e-3)
        self.Lx = cfg.get("Lx", 10e-3)
        self.Ly = cfg.get("Ly", 10e-3)
        self.n_pts = cfg.get("n_collocation", 10000)
        self.n_bc = cfg.get("n_boundary", 1000)
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.history = {"loss": []}

    def fit(self, domain_pts, bc_pts) -> dict:
        domain_pts = domain_pts.to(self.device)
        bc_pts = bc_pts.to(self.device)
        bc_ref = far_field_disp(bc_pts, SIGMA_INF, E, NU, CRACK_HALF_LENGTH).to(self.device)

        for ep in range(1, self.epochs + 1):
            self.opt.zero_grad()
            rx, ry, _, _, _ = elastic_residual(domain_pts, self.model, E, NU)
            pde_loss = (rx ** 2 + ry ** 2).mean()

            bc_pred = self.model(bc_pts)
            bc_loss = ((bc_pred - bc_ref) ** 2).mean()

            loss = self.w_pde * pde_loss + self.w_bc * bc_loss
            loss.backward()
            self.opt.step()
            self.history["loss"].append(float(loss.detach()))
            if ep % 200 == 0:
                log.info(f"PINN ep {ep}: loss={loss:.5e} pde={pde_loss:.2e} bc={bc_loss:.2e}")
        self.model.save(self.out_dir / "pinn.pt")
        return self.history
