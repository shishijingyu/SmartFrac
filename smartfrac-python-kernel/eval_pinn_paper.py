"""Produce real, trustworthy PINN-vs-Westergaard numbers for the paper.

1) Correct Westergaard Mode-I closed form (plane stress), the standard
   Z = sigma_inf * z / sqrt(z^2 - a^2) complex-potential form.
2) Train a PINN to convergence (more epochs/collocation than the 200-step demo).
3) Report relative L2 on interior points excluding the tip singularity.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from smartfrac.constants import E, NU, SIGMA_INF, CRACK_HALF_LENGTH
from smartfrac.pinn.model import PINNMLP
from smartfrac.pinn.trainer import PINNTrainer
from smartfrac.pinn.sampling import sample_domain, sample_boundary


def westergaard_correct(coords: np.ndarray, a: float, sigma_inf: float,
                        E: float, nu: float) -> dict:
    """Standard Westergaard Mode-I solution (plane stress).

    Crack along (-a, a) on x-axis, origin at crack center, remote sigma_inf.
    Returns SI units (m, Pa).
    """
    x = coords[:, 0]
    y = coords[:, 1]
    z = x + 1j * y
    # branch cut along the crack; choose principal sqrt
    Z = sigma_inf * z / np.sqrt(z ** 2 - a ** 2 + 0j)
    Zp = -sigma_inf * a ** 2 / (z ** 2 - a ** 2 + 0j) ** 1.5
    Zbar = sigma_inf * np.sqrt(z ** 2 - a ** 2 + 0j)

    sxx = Z.real - y * Zp.imag
    syy = Z.real + y * Zp.imag
    sxy = -y * Zp.real

    ux = (1.0 / E) * ((1 - nu) * Zbar.real - (1 + nu) * y * Z.imag)
    uy = (1.0 / E) * ((1 - nu) * Zbar.imag + (1 + nu) * y * Z.real)
    return {"ux": ux, "uy": uy, "sxx": sxx, "syy": syy, "sxy": sxy}


# ---- train a converged PINN ----
torch.manual_seed(0)
np.random.seed(0)
model = PINNMLP({"hidden_dims": [64, 64, 64, 64], "activation": "tanh"})
trainer = PINNTrainer(
    model,
    {"epochs": 3000, "lr": 1e-3, "n_collocation": 8000, "n_boundary": 1000,
     "Lx": 0.01, "Ly": 0.01, "tip_radius": 0.0005, "w_pde": 1.0, "w_bc": 10.0},
    out_dir="results/paper_pinn",
)
Lx = Ly = 0.01
domain = sample_domain(8000, Lx, Ly, (CRACK_HALF_LENGTH, 0.0), tip_radius=0.0005)
bc = sample_boundary(1000, Lx, Ly)
hist = trainer.fit(domain, bc)
print(f"train final loss: {hist['loss'][-1]:.4e}")

# ---- evaluate on a fresh interior grid ----
rng = np.random.default_rng(123)
n = 6000
xv = rng.uniform(-Lx, Lx, n)
yv = rng.uniform(-Ly, Ly, n)
# exclude the crack-tip singularity and the crack faces themselves for a fair,
# well-posed comparison
r_tip = np.sqrt((xv - CRACK_HALF_LENGTH) ** 2 + yv ** 2)
r_left = np.sqrt((xv + CRACK_HALF_LENGTH) ** 2 + yv ** 2)
keep = (r_tip > 3e-4) & (r_left > 3e-4)
pts = np.column_stack([xv[keep], yv[keep]]).astype(np.float32)

model.eval()
with torch.no_grad():
    u_pinn = model(torch.tensor(pts)).numpy()

ref = westergaard_correct(pts, a=CRACK_HALF_LENGTH, sigma_inf=SIGMA_INF, E=E, nu=NU)


def l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


l2x, l2y = l2(u_pinn[:, 0], ref["ux"]), l2(u_pinn[:, 1], ref["uy"])
print("=== PINN vs correct Westergaard ===")
print(f"points: {len(pts)}")
print(f"relative L2 ux: {l2x*100:.2f} %")
print(f"relative L2 uy: {l2y*100:.2f} %")
print(f"mean L2: {(l2x+l2y)/2*100:.2f} %")
print(f"PINN ux range [{u_pinn[:,0].min():.3e},{u_pinn[:,0].max():.3e}]")
print(f"Wstr ux range [{ref['ux'].min():.3e},{ref['ux'].max():.3e}]")
print(f"PINN uy range [{u_pinn[:,1].min():.3e},{u_pinn[:,1].max():.3e}]")
print(f"Wstr uy range [{ref['uy'].min():.3e},{ref['uy'].max():.3e}]")
