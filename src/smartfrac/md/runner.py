"""ASE MD 运行器（FR-2.9）。"""
from __future__ import annotations

import time
from pathlib import Path

from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units


def run_md(atoms: Atoms, nsteps: int = 10000, temperature: float = 300.0,
           timestep: float = 0.001, log_file: str | None = None) -> dict:
    """NVT Langevin MD。返回能量演化日志。"""
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = Langevin(atoms, timestep * units.fs, temperature_K=temperature,
                    friction=0.01)
    energies: list[float] = []
    times: list[float] = []
    t0 = time.time()

    def log_step():
        energies.append(float(atoms.get_potential_energy()))
        times.append(time.time() - t0)

    dyn.attach(log_step, interval=10)
    dyn.run(nsteps)
    result = {"energies": energies, "times": times,
              "n_steps": nsteps, "wall_time": time.time() - t0}
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            f.write("step,energy\n")
            for i, e in enumerate(energies):
                f.write(f"{i*10},{e}\n")
    return result
