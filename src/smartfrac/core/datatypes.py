"""核心数据结构：模块间数据流契约（接口 A/B/C/D）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AtomicSample:
    """单个原子构型样本。"""
    numbers: np.ndarray            # (N,) 元素序号
    positions: np.ndarray          # (N, 3) Å
    cell: np.ndarray               # (3, 3) Å
    energy: float | None = None
    forces: np.ndarray | None = None     # (N, 3) eV/Å
    stress: np.ndarray | None = None    # (6,) Voigt

    def __post_init__(self) -> None:
        self.numbers = np.asarray(self.numbers, dtype=int)
        self.positions = np.asarray(self.positions, dtype=float)
        self.cell = np.asarray(self.cell, dtype=float)


@dataclass
class Dataset:
    """M1 输出 / M2-A 输入（接口 A）。"""
    positions: list[np.ndarray] | np.ndarray
    numbers: list[np.ndarray] | np.ndarray
    energies: np.ndarray            # (M,)
    forces: list[np.ndarray] | np.ndarray
    descriptors: np.ndarray | None = None   # (M, D)
    cells: list[np.ndarray] | np.ndarray | None = None
    stresses: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.energies)


@dataclass
class FieldData:
    """连续介质场数据（接口 C/D）。"""
    coords: np.ndarray              # (K, 2)
    ux: np.ndarray
    uy: np.ndarray
    sxx: np.ndarray
    syy: np.ndarray
    sxy: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
