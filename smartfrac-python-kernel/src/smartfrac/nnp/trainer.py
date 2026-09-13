"""NNP 训练循环（FR-2.4/2.5/2.6）。"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from smartfrac.core.base_trainer import Trainer, TrainConfig
from .loss import nnp_loss

log = logging.getLogger(__name__)


class NNPTrainer:
    """轻量 NNP 训练器：desc/energy/forces 已预处理好的张量。"""

    def __init__(self, model, cfg: dict, out_dir: str = "results"):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.w_e = cfg.get("w_energy", 1.0)
        self.w_f = cfg.get("w_force", 10.0)
        self.epochs = cfg.get("epochs", 200)
        self.lr = cfg.get("lr", 1e-3)
        self.patience = cfg.get("early_stop_patience", 50)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.history: dict[str, list[float]] = {"loss": [], "val_loss": []}

    def fit(self, train_data: tuple, val_data: tuple) -> dict:
        tr_desc, tr_species, tr_e, tr_f = self._to_dev(train_data)
        va_desc, va_species, va_e, va_f = self._to_dev(val_data)
        best = float("inf"); no_imp = 0
        for ep in range(1, self.epochs + 1):
            self.model.train()
            self.opt.zero_grad()
            pred_per_atom = self.model(tr_desc, tr_species)
            pred_e = self._sum_per_structure(pred_per_atom, tr_species)
            m = nnp_loss(pred_e, tr_e, pred_per_atom, torch.zeros_like(pred_per_atom),
                         self.w_e, self.w_f)
            m["loss"].backward()
            self.opt.step()

            self.model.eval()
            with torch.no_grad():
                vp = self.model(va_desc, va_species)
                va_loss = torch.nn.functional.l1_loss(
                    self._sum_per_structure(vp, va_species), va_e)
            self.history["loss"].append(float(m["loss"]))
            self.history["val_loss"].append(float(va_loss))
            if ep % 20 == 0:
                log.info(f"ep {ep}: loss={m['loss']:.4f} val={va_loss:.5f}")
            if va_loss < best:
                best = va_loss; no_imp = 0
                self.model.save(self.out_dir / "best.pt")
            else:
                no_imp += 1
                if no_imp >= self.patience:
                    log.info(f"early stop @ ep {ep}"); break
        self.model.save(self.out_dir / "last.pt")
        return self.history

    def _to_dev(self, t: tuple):
        return tuple(x.to(self.device) if torch.is_tensor(x) else x for x in t)

    @staticmethod
    def _sum_per_structure(per_atom: torch.Tensor, species) -> torch.Tensor:
        # 简化：训练时每个结构一个原子（MVP），批量用单原子构型
        return per_atom
