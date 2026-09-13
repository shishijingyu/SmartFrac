"""TC-06: NNP ASE Calculator + MD 加速比验证。

1. 加载 best.pt 包装成 ASE Calculator
2. 跑 NVT MD 100 步，测单步耗时
3. 对比 LJ 势的单步耗时
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import h5py
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.units import fs

from smartfrac.nnp.descriptor import SoapDescriptor
from smartfrac.nnp.formal_model import NNPNet

BEST = ROOT / "results" / "formal_nnp" / "best.pt"
H5 = ROOT / "data" / "processed" / "alpha_fe_train.h5"


class NNPFeCalculator:
    """把训好的 NNP 包装成 ASE Calculator（鸭子类型）。"""
    def __init__(self):
        self.model = NNPNet(147, 256)
        self.model.load_state_dict(torch.load(BEST, weights_only=True))
        self.model.eval()
        self.desc = SoapDescriptor(species=[26], r_cut=4.0, n_max=6, l_max=6)
        with h5py.File(H5, "r") as f:
            desc = f["descriptors"][:]
            forces = f["forces"][:]
        self.d_mean, self.d_std = desc.mean(0), desc.std(0) + 1e-8
        self.f_mean, self.f_std = forces.mean(), forces.std() + 1e-8
        self.e_mean, self.e_std = -7.888, 0.477

    def get_potential_energy(self, atoms=None, **kw):
        return self._calc(atoms)["energy"]

    def get_forces(self, atoms=None, **kw):
        return self._calc(atoms)["forces"]

    def _calc(self, atoms):
        d = self.desc.soap.create([atoms], n_jobs=1)
        dn = (d - self.d_mean) / self.d_std
        with torch.no_grad():
            e_per_atom, f_pred = self.model(torch.tensor(dn, dtype=torch.float32))
        e_total = (float(e_per_atom.mean()) * self.e_std + self.e_mean) * len(atoms)
        f = f_pred.numpy() * self.f_std + self.f_mean
        return {"energy": e_total, "forces": f}


def make_bulk():
    """α-Fe BCC 16 原子超胞。"""
    a = 2.87  # Å
    bcc = Atoms("Fe2", cell=[a, a, a],
                scaled_positions=[[0, 0, 0], [0.5, 0.5, 5]],
                pbc=True)
    return bcc.repeat((2, 2, 2))


def run_md(calc, label, steps=100):
    at = make_bulk()
    at.calc = calc
    MaxwellBoltzmannDistribution(at, temperature_K=300)
    dt = 1.0 * fs
    dyn = VelocityVerlet(at, dt)

    # 预热 5 步
    for _ in range(5):
        dyn.step()

    t0 = time.time()
    for _ in range(steps):
        dyn.step()
    elapsed = time.time() - t0
    per_step = elapsed / steps
    print(f"  {label}: {elapsed:.2f}s / {steps}步 = {per_step*1000:.1f} ms/步")
    return per_step


def main():
    print("=== TC-06: MD 加速比验证 ===\n")
    print("体系: BCC Fe 16 原子, 300K, NVT, dt=1fs\n")

    # NNP
    print("[1/2] NNP Calculator:")
    nnp = NNPFeCalculator()
    t_nnp = run_md(nnp, "NNP (SOAP+MLP)", steps=50)

    # LJ 对比
    print("[2/2] LJ 经典势:")
    lj = LennardJones(epsilon=0.01, sigma=2.5)
    t_lj = run_md(lj, "LennardJones", steps=100)

    print(f"\n=== 结果 ===")
    print(f"NNP 单步:   {t_nnp*1000:.1f} ms")
    print(f"LJ 单步:    {t_lj*1000:.1f} ms")
    print(f"NNP/LJ 比:  {t_nnp/t_lj:.1f}×")
    print(f"\n理论 DFT 单步 (VASP): ~10000 ms (10s/步)")
    print(f"NNP 相对 DFT 加速:   {10000/(t_nnp*1000):.0f}×")


if __name__ == "__main__":
    main()
