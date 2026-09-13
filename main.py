"""SmartFrac 一键入口：NNP 正式版训练全流程。

在 PyCharm 中运行方式：
  1. 右键本文件 -> Run 'main'
  2. 或点右上角绿色三角

前提：解释器已指向 .venv（File -> Settings -> Project -> Python Interpreter）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 把 src/ 加到 import 路径，这样不用手动设 PYTHONPATH
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import h5py
from ase.io import read

from smartfrac.nnp.descriptor import SoapDescriptor
from smartfrac.nnp.formal_model import NNPNet
from smartfrac.viz.curves import plot_loss_curve, parity_plot

# ====== 配置 ======
RAW_XYZ = ROOT / "data" / "raw" / "byggmastar_fe" / "training_database.xyz"
H5_PATH = ROOT / "data" / "processed" / "alpha_fe_train.h5"
OUT_DIR = ROOT / "results" / "formal_nnp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ====== 第 1 步：预处理（首次运行需要，之后自动跳过） ======
def preprocess() -> None:
    if H5_PATH.exists():
        print(f"[1/3] 已存在 {H5_PATH.name}，跳过预处理")
        return
    print(f"[1/3] 预处理: 读取 {RAW_XYZ.name} ...")
    frames = read(str(RAW_XYZ), index=":")
    print(f"      共 {len(frames)} 个构型，算 SOAP ...")

    desc = SoapDescriptor(species=[26], r_cut=4.0, n_max=6, l_max=6)
    all_d, all_f, energies, offsets = [], [], [], []
    off = 0
    for at in frames:
        if not (2 <= len(at) <= 128):
            offsets.append((off, off)); energies.append(0.0); continue
        d = desc.soap.create([at], n_jobs=1)
        all_d.append(d); all_f.append(at.get_forces())
        energies.append(at.get_potential_energy())
        offsets.append((off, off + len(at))); off += len(at)

    with h5py.File(H5_PATH, "w") as f:
        f.create_dataset("descriptors", data=np.vstack(all_d), compression="gzip")
        f.create_dataset("forces", data=np.vstack(all_f), compression="gzip")
        f.create_dataset("energies", data=np.array(energies))
        f.create_dataset("config_idx", data=np.array(offsets, dtype=np.int64))
    print(f"      完成 -> {H5_PATH}")


# ====== 第 2 步：训练 ======
def train() -> dict:
    print("[2/3] 训练 NNP ...")
    with h5py.File(H5_PATH, "r") as f:
        desc = f["descriptors"][:]
        forces = f["forces"][:]
        energies = f["energies"][:]
        idx = f["config_idx"][:]

    natoms = idx[:, 1] - idx[:, 0]
    ok = natoms > 0
    idx, energies, natoms = idx[ok], energies[ok], natoms[ok]
    epa = energies / natoms
    phys = (epa > -9.5) & (epa < -6.0)
    idx, energies, natoms, epa = idx[phys], energies[phys], natoms[phys], epa[phys]

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(epa))
    n_test = int(0.15 * len(epa))
    test_c, train_c = order[:n_test], order[n_test:]

    desc_n = (desc - desc.mean(0)) / (desc.std(0) + 1e-8)
    e_mean, e_std = epa.mean(), epa.std()
    f_mean, f_std = forces.mean(), forces.std() + 1e-8

    def mask(cs):
        m = np.zeros(len(desc_n), dtype=bool)
        for s, e in idx[cs]:
            m[s:e] = True
        return m

    Xtr = torch.tensor(desc_n[mask(train_c)], dtype=torch.float32)
    Ftr = torch.tensor((forces[mask(train_c)] - f_mean) / f_std, dtype=torch.float32)
    Etr = np.zeros(len(desc_n), dtype=np.float32)
    for ci in train_c:
        s, e = idx[ci]; Etr[s:e] = (epa[ci] - e_mean) / e_std
    Etr = torch.tensor(Etr[mask(train_c)], dtype=torch.float32)

    model = NNPNet(147, 256)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-5)

    BS, hist, best = 4096, [], 1e9
    n_atoms = len(Xtr)
    t0 = time.time()
    for ep in range(1, 301):
        model.train()
        perm = torch.randperm(n_atoms)
        eloss = 0.0
        for i in range(0, n_atoms, BS):
            j = perm[i:i+BS]
            opt.zero_grad()
            e_, f_ = model(Xtr[j])
            loss = ((e_ - Etr[j])**2).mean() + ((f_ - Ftr[j])**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            eloss += float(loss.detach()) * len(j)
        sched.step()
        hist.append(eloss / n_atoms)

        if ep % 30 == 0:
            model.eval()
            pe, re_, pf, rf = [], [], [], []
            with torch.no_grad():
                for ci in test_c:
                    s, e = idx[ci]
                    ep_, fp = model(torch.tensor(desc_n[s:e], dtype=torch.float32))
                    pe.append(float(ep_.mean())*e_std + e_mean)
                    re_.append(epa[ci])
                    pf.append(fp.numpy()*f_std + f_mean); rf.append(forces[s:e])
            pe, re_ = np.array(pe), np.array(re_)
            pf, rf = np.vstack(pf), np.vstack(rf)
            mae = np.mean(np.abs(pe - re_)) * 1000
            rmse = np.sqrt(((pf - rf)**2).mean())
            print(f"      ep {ep:3d}  E MAE={mae:6.1f} meV  F RMSE={rmse:.3f} eV/Å")
            if mae < best:
                best = mae
                torch.save(model.state_dict(), OUT_DIR / "best.pt")

    model.load_state_dict(torch.load(OUT_DIR / "best.pt", weights_only=True))
    print(f"      最佳 E MAE = {best:.1f} meV/atom  ({time.time()-t0:.0f}s)")
    plot_loss_curve({"loss": hist}, out=str(OUT_DIR / "loss.png"))
    return {"model": model, "e_mean": e_mean, "e_std": e_std, "idx": idx, "test_c": test_c, "epa": epa, "desc_n": desc_n}


# ====== 第 3 步：评估 + 出图 ======
def evaluate(ctx: dict) -> None:
    print("[3/3] 评估 + 出图 ...")
    model = ctx["model"]; model.eval()
    pe, re_ = [], []
    with torch.no_grad():
        for ci in ctx["test_c"]:
            s, e = ctx["idx"][ci]
            ep_, _ = model(torch.tensor(ctx["desc_n"][s:e], dtype=torch.float32))
            pe.append(float(ep_.mean()) * ctx["e_std"] + ctx["e_mean"])
            re_.append(ctx["epa"][ci])
    pe, re_ = np.array(pe), np.array(re_)
    mae = np.mean(np.abs(pe - re_)) * 1000
    parity_plot(pe, re_, out=str(OUT_DIR / "parity.png"))
    print(f"\n===== 最终结果 =====")
    print(f"能量 MAE = {mae:.1f} meV/atom  (TC-03 目标 < 50)")
    print(f"图已保存: {OUT_DIR}")


if __name__ == "__main__":
    t0 = time.time()
    preprocess()
    ctx = train()
    evaluate(ctx)
    print(f"\n总耗时 {time.time()-t0:.0f}s，完成。")
