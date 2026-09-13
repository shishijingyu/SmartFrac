"""归一化（FR-1.6/1.7）：坐标 min-max，能量/力 z-score，可逆。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Normalizer:
    """能量/力标签归一化参数，训练后 inverse 还原。"""
    energy_mean: float = 0.0
    energy_std: float = 1.0
    force_std: float = 1.0
    cell_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cell_max: np.ndarray = field(default_factory=lambda: np.ones(3))
    fitted: bool = False

    def fit_labels(self, energies: np.ndarray, forces: np.ndarray) -> None:
        self.energy_mean = float(np.mean(energies))
        self.energy_std = float(np.std(energies) + 1e-12)
        self.force_std = float(np.std(forces) + 1e-12)
        self.fitted = True

    def transform_energy(self, e: np.ndarray) -> np.ndarray:
        return (e - self.energy_mean) / self.energy_std

    def inverse_energy(self, e: np.ndarray) -> np.ndarray:
        return e * self.energy_std + self.energy_mean

    def transform_force(self, f: np.ndarray) -> np.ndarray:
        return f / self.force_std

    def inverse_force(self, f: np.ndarray) -> np.ndarray:
        return f * self.force_std

    def fit_coords(self, positions_list: list[np.ndarray]) -> None:
        allp = np.vstack(positions_list)
        self.cell_min = allp.min(axis=0)
        self.cell_max = allp.max(axis=0)

    def transform_coords(self, p: np.ndarray) -> np.ndarray:
        return (p - self.cell_min) / (self.cell_max - self.cell_min + 1e-12)

    def inverse_coords(self, p: np.ndarray) -> np.ndarray:
        return p * (self.cell_max - self.cell_min) + self.cell_min

    def to_dict(self) -> dict:
        return {
            "energy_mean": self.energy_mean, "energy_std": self.energy_std,
            "force_std": self.force_std,
            "cell_min": self.cell_min.tolist(), "cell_max": self.cell_max.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        n = cls()
        n.energy_mean = d["energy_mean"]; n.energy_std = d["energy_std"]
        n.force_std = d["force_std"]
        n.cell_min = np.array(d["cell_min"]); n.cell_max = np.array(d["cell_max"])
        n.fitted = True
        return n
