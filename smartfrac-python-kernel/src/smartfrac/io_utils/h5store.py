"""HDF5 数据集读写（FR-1.11）。"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from smartfrac.core.datatypes import AtomicSample


def save_dataset(path: str | Path, samples: list[AtomicSample],
                 descriptors: np.ndarray | None = None,
                 metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    energies = np.array([s.energy or 0.0 for s in samples])
    forces = np.array([s.forces if s.forces is not None else np.zeros_like(s.positions)
                       for s in samples])
    numbers = np.array([s.numbers for s in samples], dtype=object)
    positions = np.array([s.positions for s in samples], dtype=object)
    with h5py.File(path, "w") as f:
        f.create_dataset("energies", data=energies)
        f.create_dataset("forces", data=forces)
        f.create_dataset("positions", data=np.array(positions, dtype=object))
        f.create_dataset("numbers", data=np.array(numbers, dtype=object))
        if descriptors is not None:
            f.create_dataset("descriptors", data=descriptors)
        f.attrs["metadata"] = json.dumps(metadata or {})


def load_dataset(path: str | Path) -> tuple[list[AtomicSample], np.ndarray | None, dict]:
    with h5py.File(path, "r") as f:
        energies = f["energies"][:]
        forces = f["forces"][:]
        positions = f["positions"][:].tolist()
        numbers = f["numbers"][:].tolist()
        desc = f["descriptors"][:] if "descriptors" in f else None
        meta = json.loads(f.attrs.get("metadata", "{}"))
    samples = [AtomicSample(numbers=np.array(n, dtype=int),
                            positions=np.array(p),
                            cell=np.eye(3),
                            energy=float(e), forces=np.array(fo))
               for n, p, e, fo in zip(numbers, positions, energies, forces)]
    return samples, desc, meta
