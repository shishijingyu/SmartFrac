"""模型统一基类（NFR-1）：NNP / PINN 都继承它。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """所有 AI 模型的公共接口：save/load/config。"""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.cfg: dict[str, Any] = cfg or {}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    # ---- 序列化 ----
    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        payload = {"state_dict": self.state_dict(), "cfg": self.cfg}
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "BaseModel":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        obj = cls(cfg=payload.get("cfg", {}))
        obj.load_state_dict(payload["state_dict"])
        obj.eval()
        return obj
