"""数据集划分（FR-1.9）：按能量区间分层 70/15/15。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def stratified_split(energies: np.ndarray, ratios=(0.7, 0.15, 0.15),
                     seed: int = 42, n_bins: int = 5) -> SplitResult:
    rng = np.random.default_rng(seed)
    # 按能量分桶，桶内分层抽样
    bins = np.quantile(energies, np.linspace(0, 1, n_bins + 1))
    bin_id = np.digitize(energies, bins[1:-1])
    train, val, test = [], [], []
    for b in np.unique(bin_id):
        idx = np.where(bin_id == b)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        train.extend(idx[:n_tr])
        val.extend(idx[n_tr:n_tr + n_va])
        test.extend(idx[n_tr + n_va:])
    return SplitResult(np.array(train), np.array(val), np.array(test))
