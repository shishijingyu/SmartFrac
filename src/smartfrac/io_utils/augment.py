"""样本增广（FR-1.12）：±0.05Å 坐标随机扰动。"""
from __future__ import annotations

import numpy as np

from smartfrac.core.datatypes import AtomicSample


def augment(samples: list[AtomicSample], n_aug: int = 1,
            noise: float = 0.05, seed: int = 42) -> list[AtomicSample]:
    rng = np.random.default_rng(seed)
    out: list[AtomicSample] = list(samples)
    for _ in range(n_aug):
        for s in samples:
            new_pos = s.positions + rng.normal(0, noise, size=s.positions.shape)
            out.append(AtomicSample(
                numbers=s.numbers.copy(), positions=new_pos, cell=s.cell.copy(),
                energy=s.energy, forces=s.forces, stress=s.stress))
    return out
