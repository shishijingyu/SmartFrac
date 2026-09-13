"""预处理 Byggmästar 数据集：全部构型算 SOAP 描述符，存 HDF5。

输出: data/processed/alpha_fe_train.h5
  descriptors: (total_atoms, 147)  所有原子的 SOAP 拼平
  forces:      (total_atoms, 3)    所有原子的 DFT 力
  energies:    (n_configs,)         每构型总能量
  natoms:      (n_configs,)         每构型原子数
  config_idx:  (n_configs, 2)       每构型在拼平数组中的 [start, end)
"""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
from ase.io import read

from smartfrac.nnp.descriptor import SoapDescriptor

SRC = Path("data/raw/byggmastar_fe/training_database.xyz")
OUT = Path("data/processed/alpha_fe_train.h5")
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    print(f"读取 {SRC} ...")
    frames = read(str(SRC), index=":")
    print(f"共 {len(frames)} 个构型")

    desc = SoapDescriptor(species=[26], r_cut=4.0, n_max=6, l_max=6)
    print(f"SOAP 维度: {desc.dim}")

    all_desc, all_forces = [], []
    energies, natoms, offsets = [], [], []
    offset = 0

    for i, at in enumerate(frames):
        if len(at) < 2 or len(at) > 128:
            # 记录偏移但跳过太大/太小
            offsets.append((offset, offset))
            energies.append(0.0)
            natoms.append(len(at))
            continue
        d = desc.soap.create([at], n_jobs=1)  # (N, D)
        all_desc.append(d)
        all_forces.append(at.get_forces())
        energies.append(at.get_potential_energy())
        natoms.append(len(at))
        offsets.append((offset, offset + len(at)))
        offset += len(at)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(frames)}, {time.time()-t0:.1f}s")

    descriptors = np.vstack(all_desc)
    forces = np.vstack(all_forces)
    energies = np.array(energies, dtype=np.float64)
    natoms = np.array(natoms, dtype=np.int32)
    config_idx = np.array(offsets, dtype=np.int64)

    print(f"总原子数: {descriptors.shape[0]}, 总构型: {len(energies)}")

    with h5py.File(OUT, "w") as f:
        f.create_dataset("descriptors", data=descriptors, compression="gzip")
        f.create_dataset("forces", data=forces, compression="gzip")
        f.create_dataset("energies", data=energies)
        f.create_dataset("natoms", data=natoms)
        f.create_dataset("config_idx", data=config_idx)
        f.attrs["soap_dim"] = desc.dim
        f.attrs["n_configs"] = len(energies)

    print(f"已存 -> {OUT}  ({time.time()-t0:.1f}s)")
    print(f"文件大小: {OUT.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
