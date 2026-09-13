"""Evaluate the trained formal NNP checkpoint on the held-out test split.

Reproduces exactly the preprocessing / split / normalization used in
main.py and examples/02c_train_nnp_formal.py so that reported metrics
match the training-time test set.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import h5py
import numpy as np
import torch
from smartfrac.nnp.formal_model import NNPNet

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

H5 = ROOT / "data" / "processed" / "alpha_fe_train.h5"
CKPT = ROOT / "results" / "formal_nnp" / "best.pt"

with h5py.File(H5, "r") as f:
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

model = NNPNet(147, 256)
model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
model.eval()

pe, re_ = [], []
pf, rf = [], []
with torch.no_grad():
    for ci in test_c:
        s, e = idx[ci]
        ep_, fp = model(torch.tensor(desc_n[s:e], dtype=torch.float32))
        pe.append(float(ep_.mean()) * e_std + e_mean)
        re_.append(epa[ci])
        pf.append(fp.numpy() * f_std + f_mean)
        rf.append(forces[s:e])

pe, re_ = np.array(pe), np.array(re_)
pf, rf = np.vstack(pf), np.vstack(rf)
e_mae = float(np.mean(np.abs(pe - re_)) * 1000.0)        # meV/atom
e_rmse = float(np.sqrt(np.mean((pe - re_) ** 2)) * 1000.0)
f_rmse = float(np.sqrt(np.mean((pf - rf) ** 2)))         # eV/A
f_mae = float(np.mean(np.abs(pf - rf)))

# correlation coefficient
corr = float(np.corrcoef(pe, re_)[0, 1])

print("=== NNP formal checkpoint evaluation ===")
print(f"train configs : {len(train_c)}")
print(f"test  configs : {len(test_c)}")
print(f"total kept configs: {len(epa)}")
print(f"energy MAE   : {e_mae:.2f} meV/atom")
print(f"energy RMSE  : {e_rmse:.2f} meV/atom")
print(f"force  RMSE  : {f_rmse:.4f} eV/A")
print(f"force  MAE   : {f_mae:.4f} eV/A")
print(f"energy Pearson r: {corr:.4f}")
print(f"e_mean={e_mean:.4f} eV, e_std={e_std:.4f} eV")
