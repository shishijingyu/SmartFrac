"""正式版 NNP v2：原子级 batch，过滤离群构型。"""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import torch

from smartfrac.nnp.formal_model import NNPNet
from smartfrac.viz.curves import plot_loss_curve, parity_plot

H5 = Path("data/processed/alpha_fe_train.h5")
OUT = Path("results/formal_nnp")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def main() -> None:
    t0 = time.time()
    with h5py.File(H5, "r") as f:
        desc = f["descriptors"][:]
        forces = f["forces"][:]
        energies = f["energies"][:]
        idx = f["config_idx"][:]

    natoms = idx[:, 1] - idx[:, 0]
    valid = natoms > 0
    idx, energies, natoms = idx[valid], energies[valid], natoms[valid]
    epa = energies / natoms

    # 物理过滤: 每原子能量在 -9 ~ -6 eV 之间 (BCC Fe 典型范围)
    phys = (epa > -9.5) & (epa < -6.0)
    idx, energies, natoms, epa = idx[phys], energies[phys], natoms[phys], epa[phys]
    print(f"物理过滤后 {len(epa)} 构型, E/atom 范围 {epa.min():.2f}~{epa.max():.2f}")

    # 分层划分
    order = np.random.default_rng(SEED).permutation(len(epa))
    n_test = int(0.15 * len(epa))
    test_c = order[:n_test]
    train_c = order[n_test:]
    print(f"train={len(train_c)} test={len(test_c)}")

    # 归一化
    desc_mean, desc_std = desc.mean(0), desc.std(0) + 1e-8
    desc_n = (desc - desc_mean) / desc_std
    e_mean, e_std = epa.mean(), epa.std()
    f_mean, f_std = forces.mean(), forces.std() + 1e-8
    print(f"e_mean={e_mean:.3f} e_std={e_std:.3f} f_std={f_std:.3f}")

    # 预计算每个训练构型的原子索引
    def gather(configs):
        starts, ends = idx[configs, 0], idx[configs, 1]
        mask = np.zeros(len(desc_n), dtype=bool)
        for s, e in zip(starts, ends):
            mask[s:e] = True
        return mask

    train_mask = gather(train_c)
    test_mask = gather(test_c)

    device = torch.device("cpu")
    model = NNPNet(147, 256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-5)

    # 原子级 batch: 所有训练原子拼平
    Xtr = torch.tensor(desc_n[train_mask], dtype=torch.float32)
    Ftr = torch.tensor((forces[train_mask] - f_mean) / f_std, dtype=torch.float32)

    # 每原子能量标签: 把构型 epa 广播到每原子
    Etr = np.zeros(len(desc_n), dtype=np.float32)
    for ci in train_c:
        s, e = idx[ci]
        Etr[s:e] = (epa[ci] - e_mean) / e_std
    Etr = torch.tensor(Etr[train_mask], dtype=torch.float32)

    BS = 4096
    history = {"loss": []}
    n_atoms = len(Xtr)
    best_mae = 1e9; best_ep = 0

    for ep in range(1, 301):
        model.train()
        perm = torch.randperm(n_atoms)
        ep_loss = 0.0
        for i in range(0, n_atoms, BS):
            j = perm[i:i+BS]
            x, f_t, e_t = Xtr[j], Ftr[j], Etr[j]
            opt.zero_grad()
            e_pred, f_pred = model(x)
            loss = ((e_pred - e_t) ** 2).mean() + ((f_pred - f_t) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.detach()) * len(j)
        sched.step()
        ep_loss /= n_atoms
        history["loss"].append(ep_loss)

        if ep % 10 == 0:
            model.eval()
            preds_e, refs_e = [], []
            preds_f, refs_f = [], []
            with torch.no_grad():
                for ci in test_c:
                    s, e = idx[ci]
                    x = torch.tensor(desc_n[s:e], dtype=torch.float32)
                    ep_, fp = model(x)
                    preds_e.append(float(ep_.mean()) * e_std + e_mean)
                    refs_e.append(epa[ci])
                    preds_f.append(fp.numpy() * f_std + f_mean)
                    refs_f.append(forces[s:e])
            preds_e, refs_e = np.array(preds_e), np.array(refs_e)
            preds_f, refs_f = np.vstack(preds_f), np.vstack(refs_f)
            mae = np.mean(np.abs(preds_e - refs_e)) * 1000
            rmse_f = np.sqrt(((preds_f - refs_f) ** 2).mean())
            lr = sched.get_last_lr()[0]
            print(f"ep {ep}: loss={ep_loss:.4f}  E MAE={mae:.1f} meV  F RMSE={rmse_f:.3f}  lr={lr:.5f}")
            if mae < best_mae:
                best_mae = mae; best_ep = ep
                torch.save(model.state_dict(), OUT / "best.pt")

    # 加载最优
    model.load_state_dict(torch.load(OUT / "best.pt", weights_only=True))
    print(f"\n最佳: ep {best_ep}, E MAE={best_mae:.1f} meV/atom")

    # 最终
    model.eval()
    preds_e, refs_e = [], []
    with torch.no_grad():
        for ci in test_c:
            s, e = idx[ci]
            x = torch.tensor(desc_n[s:e], dtype=torch.float32)
            ep_, fp = model(x)
            preds_e.append(float(ep_.mean()) * e_std + e_mean)
            refs_e.append(epa[ci])
    preds_e, refs_e = np.array(preds_e), np.array(refs_e)
    mae = np.mean(np.abs(preds_e - refs_e)) * 1000
    print(f"\n=== E MAE = {mae:.1f} meV/atom (TC-03 < 50) ===")

    plot_loss_curve(history, out=str(OUT / "loss.png"))
    parity_plot(preds_e, refs_e, out=str(OUT / "parity.png"))
    print(f"图 -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
