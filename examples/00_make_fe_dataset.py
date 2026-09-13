"""示例 0：生成 α-Fe (BCC) 训练构型。

注意：
- 原子坐标是物理真实的 BCC-Fe 超胞（各种畸变）；
- 能量/力标签当前用 ASE 内置的 LennardJones 势打标签——这是**占位标签**，
  仅用于打通"生成→SOAP→NNP 训练"软件管线；
- 拿到真 DFT 数据后，只需把 get_labels 函数换成读 VASP/MP 输出即可。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell
from ase.calculators.lj import LennardJones
from ase.io import write

OUT = Path("data/raw/alpha_fe")
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
calc = LennardJones(epsilon=0.1, sigma=2.5, rc=10.0)


def tag(at: Atoms) -> Atoms:
    at.calc = calc
    return at


def make_perturbed(n_per_temp: int = 10) -> list[Atoms]:
    """不同温度的热扰动 BCC 超胞。"""
    out = []
    for temp in [100, 300, 600, 900]:
        for _ in range(n_per_temp):
            at = bulk("Fe", "bcc", a=2.8553).repeat((4, 4, 4))
            # 模拟热扰动：坐标加小噪声
            noise = np.sqrt(temp / 300) * 0.03
            at.positions += rng.normal(0, noise, size=at.positions.shape)
            out.append(tag(at))
    return out


def make_strained(n_per_axis: int = 8) -> list[Atoms]:
    """单轴应变（拉伸/压缩 ±5%）。"""
    out = []
    base = bulk("Fe", "bcc", a=2.8553).repeat((3, 3, 3))
    for strain in np.linspace(-0.05, 0.05, 11):
        for _ in range(n_per_axis // 11 + 1):
            at = base.copy()
            cell = at.cell.copy()
            cell[0, :] *= (1 + strain)
            at.set_cell(cell, scale_atoms=True)
            out.append(tag(at))
    return out


def make_vacancy(n: int = 10) -> list[Atoms]:
    """移除 1 个原子形成空位。"""
    out = []
    for _ in range(n):
        at = bulk("Fe", "bcc", a=2.8553).repeat((4, 4, 4))
        i = rng.integers(0, len(at))
        del at[i]
        out.append(tag(at))
    return out


if __name__ == "__main__":
    all_frames = make_perturbed() + make_strained() + make_vacancy()
    out_file = OUT / "alpha_fe_dataset.xyz"
    write(str(out_file), all_frames)
    energies = np.array([a.get_potential_energy() for a in all_frames])
    print(f"已生成 {len(all_frames)} 个 α-Fe 构型 -> {out_file}")
    print(f"能量范围: {energies.min():.3f} ~ {energies.max():.3f} eV")
    print(f"原子数: {len(all_frames[0])} / frame")
    print()
    print("注意: 当前标签来自 LJ 占位势，非 DFT。")
    print("拿到真 DFT 数据后，替换 tag() 函数即可。")
