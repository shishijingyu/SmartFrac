"""示例 2：用 examples/00 生成的 Fe 构型跑 NNP 训练最小 demo。

数据来源: data/raw/alpha_fe/alpha_fe_dataset.xyz (61 个 BCC-Fe 构型, LJ 占位标签)
真 DFT 数据接入后, 只需把 calc 换成读 VASP 输出。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase.io import read
from sklearn.model_selection import train_test_split

from smartfrac.nnp.descriptor import SoapDescriptor
from smartfrac.nnp.model import ElementEnergyNet
from smartfrac.nnp.loss import nnp_loss
from smartfrac.io_utils.split import stratified_split
from smartfrac.viz.curves import plot_loss_curve, parity_plot

DATA = Path("data/raw/alpha_fe/alpha_fe_dataset.xyz")
OUT = Path("results/demo_nnp")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def main() -> None:
    frames = read(str(DATA), index=":")
    print(f"读取 {len(frames)} 个构型")

    # 读 xyz 后 calculator 不保留, 重新挂 LJ
    from ase.calculators.lj import LennardJones
    lj = LennardJones(epsilon=0.1, sigma=2.5, rc=10.0)
    for at in frames:
        at.calc = lj

    # 1. SOAP 描述符
    species = [26]
    desc_calc = SoapDescriptor(species=species, r_cut=4.0, n_max=6, l_max=6)
    print(f"SOAP 维度: {desc_calc.dim}")

    descriptors, energies, forces_flat = [], [], []
    for at in frames:
        d = desc_calc.soap.create([at], n_jobs=1)[0]
        descriptors.append(d)
        energies.append(at.get_potential_energy())
        forces_flat.append(at.get_forces().mean())
    descriptors = np.vstack(descriptors)
    energies = np.array(energies)
    print(f"描述符形状: {descriptors.shape}, 能量范围: {energies.min():.2f}~{energies.max():.2f} eV")

    # 2. 划分
    sp = stratified_split(energies)
    Xtr, Xte = descriptors[sp.train_idx], descriptors[sp.test_idx]
    ytr, yte = energies[sp.train_idx], energies[sp.test_idx]
    print(f"train={len(Xtr)} test={len(Xte)}")

    # 3. 模型
    model = ElementEnergyNet({
        "descriptor_dim": desc_calc.dim,
        "hidden_dims": [64, 64, 32],
        "activation": "tanh",
        "species": species,
    })
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32)
    sp_tr = torch.full((len(Xtr_t),), 26, dtype=torch.long)
    sp_te = torch.full((len(Xte_t),), 26, dtype=torch.long)

    # 4. 训练
    history = {"loss": []}
    for ep in range(1, 301):
        model.train()
        opt.zero_grad()
        pred = model(Xtr_t, sp_tr).sum()  # 单构型总能量 = 原子贡献和
        loss = torch.nn.functional.l1_loss(pred.reshape(1), ytr_t.mean().reshape(1))
        # 简化: 用每构型平均能量做监督
        pred_mean = model(Xtr_t, sp_tr).mean()
        loss = torch.nn.functional.mse_loss(pred_mean.reshape(1), ytr_t.mean().reshape(1))
        loss.backward(); opt.step()
        history["loss"].append(float(loss))
        if ep % 50 == 0:
            print(f"ep {ep}: loss={loss:.4f}")

    # 5. 评估
    model.eval()
    with torch.no_grad():
        pred_test = model(Xte_t, sp_te).numpy()
    ref_test = yte

    # 6. 画图
    plot_loss_curve(history, out=str(OUT / "loss.png"))
    parity_plot(pred_test, ref_test, out=str(OUT / "parity.png"))

    mae = float(np.mean(np.abs(pred_test - ref_test)))
    print(f"\nNNP demo 完成 -> {OUT}")
    print(f"测试集能量 MAE = {mae:.4f} eV (占位 LJ 标签, 不代表 DFT 精度)")


if __name__ == "__main__":
    main()
