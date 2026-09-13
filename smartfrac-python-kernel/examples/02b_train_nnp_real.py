"""示例 2b：用 Byggmästar 真 DFT 数据集训练 NNP。

数据: data/raw/byggmastar_fe/training_database.xyz
  - 18237 个 α-Fe 构型 (BCC→FCC bain路径, 液态, 缺陷)
  - 每构型含 DFT energy + forces
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from smartfrac.nnp.descriptor import SoapDescriptor
from smartfrac.nnp.model import ElementEnergyNet
from smartfrac.io_utils.split import stratified_split
from smartfrac.viz.curves import plot_loss_curve, parity_plot

DATA = Path("data/raw/byggmastar_fe/training_database.xyz")
OUT = Path("results/real_nnp")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MAX_FRAMES = 2000  # 先用前 2000 个, CPU 上可承受


def main() -> None:
    t0 = time.time()
    print(f"读取 {DATA} ...")
    frames = read(str(DATA), index=f":{MAX_FRAMES}")
    print(f"读取 {len(frames)} 个构型, {time.time()-t0:.1f}s")

    # 过滤: 只留 2~100 原子的小体系
    frames = [a for a in frames if 2 <= len(a) <= 64]
    print(f"过滤后 {len(frames)} 个")

    desc_calc = SoapDescriptor(species=[26], r_cut=4.0, n_max=6, l_max=6)
    print(f"SOAP 维度: {desc_calc.dim}")

    # 对每个构型算: 每原子描述符 + 总能量 + 每原子平均力
    energies, per_atom_feats, natoms = [], [], []
    for i, at in enumerate(frames):
        feats = desc_calc.soap.create([at], n_jobs=1)  # (N, D)
        per_atom_feats.append(feats)
        energies.append(at.get_potential_energy())
        natoms.append(len(at))
        if (i + 1) % 200 == 0:
            print(f"  SOAP {i+1}/{len(frames)}, {time.time()-t0:.1f}s")

    # 每构型聚合: 描述符平均 + 总能量
    desc_mean = np.array([f.mean(axis=0) for f in per_atom_feats])
    energies = np.array(energies)
    natoms = np.array(natoms)
    e_per_atom = energies / natoms
    print(f"每原子能量范围: {e_per_atom.min():.3f} ~ {e_per_atom.max():.3f} eV/atom")

    # 划分
    sp = stratified_split(e_per_atom, seed=SEED)
    Xtr, Xte = desc_mean[sp.train_idx], desc_mean[sp.test_idx]
    ytr, yte = e_per_atom[sp.train_idx], e_per_atom[sp.test_idx]
    print(f"train={len(Xtr)} test={len(Xte)}")

    # 模型: 把每原子平均描述符映射到每原子能量
    model = torch.nn.Sequential(
        torch.nn.Linear(desc_calc.dim, 128), torch.nn.Tanh(),
        torch.nn.Linear(128, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 1),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).reshape(-1, 1)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32).reshape(-1, 1)

    history = {"loss": []}
    for ep in range(1, 501):
        model.train()
        opt.zero_grad()
        pred = model(Xtr_t)
        loss = torch.nn.functional.mse_loss(pred, ytr_t)
        loss.backward(); opt.step()
        history["loss"].append(float(loss))
        if ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                vloss = torch.nn.functional.mse_loss(model(Xte_t), yte_t)
            print(f"ep {ep}: train={loss:.4f} test={vloss:.4f}")

    # 评估
    model.eval()
    with torch.no_grad():
        pred_test = model(Xte_t).numpy().flatten()
    mae_meV = float(np.mean(np.abs(pred_test - yte)) * 1000)
    print(f"\n测试集 每原子能量 MAE = {mae_meV:.1f} meV/atom  (TC-03 通过线: <50)")

    plot_loss_curve(history, out=str(OUT / "loss.png"))
    parity_plot(pred_test, yte, out=str(OUT / "parity.png"))
    print(f"图 -> {OUT}")


if __name__ == "__main__":
    main()
