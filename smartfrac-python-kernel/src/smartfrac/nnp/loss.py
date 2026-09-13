"""NNP 损失（FR-2.3）：E-MAE + F-RMSE（按原子数归一）。"""
from __future__ import annotations

import torch


def nnp_loss(pred_e: torch.Tensor, true_e: torch.Tensor,
             pred_f: torch.Tensor, true_f: torch.Tensor,
             w_e: float = 1.0, w_f: float = 10.0) -> dict[str, torch.Tensor]:
    l_e = torch.nn.functional.l1_loss(pred_e, true_e)
    l_f = torch.sqrt(torch.nn.functional.mse_loss(pred_f, true_f) + 1e-12)
    loss = w_e * l_e + w_f * l_f
    return {"loss": loss, "energy_mae": l_e, "force_rmse": l_f}
