"""Trainer 模板：训练循环、早停、checkpoint、TensorBoard（FR-2.4/2.5/2.6）。"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "plateau"
    early_stop_patience: int = 50
    device: str = "auto"
    out_dir: str = "results"


@dataclass
class Trainer:
    model: torch.nn.Module
    cfg: TrainConfig
    train_step: Callable[[Any], dict[str, float]]
    val_step: Callable[[Any], dict[str, float]]
    train_loader: Any = None
    val_loader: Any = None
    history: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.device = self._resolve_device(self.cfg.device)
        self.model.to(self.device)
        self.opt = self._make_optimizer()
        self.sched = self._make_scheduler()
        self.writer = SummaryWriter(Path(self.cfg.out_dir) / "tensorboard")
        self.best_val = float("inf")
        self.best_epoch = -1
        self.no_improve = 0

    @staticmethod
    def _resolve_device(mode: str) -> torch.device:
        if mode == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(mode)

    def _make_optimizer(self) -> torch.optim.Optimizer:
        if self.cfg.optimizer.lower() == "adamw":
            return torch.optim.AdamW(self.model.parameters(),
                                     lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def _make_scheduler(self) -> Any:
        if self.cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt, mode="min", factor=0.5, patience=20)
        return None

    # ---- 主循环 ----
    def fit(self) -> dict[str, list[float]]:
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_metrics = self.train_step(self.train_loader)
            val_metrics = self.val_step(self.val_loader)
            self._record(epoch, train_metrics, val_metrics)

            val_loss = val_metrics.get("val_loss", float("inf"))
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.best_epoch = epoch
                self.no_improve = 0
                self.model.save(Path(self.cfg.out_dir) / "best.pt")
            else:
                self.no_improve += 1

            if self.sched is not None:
                self.sched.step(val_loss)

            log.info(f"epoch {epoch:4d} | train {train_metrics} | val {val_metrics} "
                     f"| {time.time()-t0:.1f}s")

            if self.no_improve >= self.cfg.early_stop_patience:
                log.info(f"early stop @ epoch {epoch}, best={self.best_val:.5f}")
                break

        self.model.save(Path(self.cfg.out_dir) / "last.pt")
        self.writer.close()
        return self.history

    def _record(self, epoch: int, tr: dict, va: dict) -> None:
        for k, v in {**{f"train_{kk}": vv for kk, vv in tr.items()},
                     **{f"val_{kk}": vv for kk, vv in va.items()}}.items():
            self.history.setdefault(k, []).append(v)
            self.writer.add_scalar(k, v, epoch)
