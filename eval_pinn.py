"""Evaluate the trained PINN checkpoint against the Westergaard closed form.

The PINN was trained with far-field Dirichlet boundary conditions on a
square domain with a central crack; here we sample interior points,
predict (ux, uy), and compare with the Westergaard Mode-I reference.
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
from smartfrac.pinn.benchmarks.westergaard import westergaard_i, l2_relative

CKPT = ROOT / "results" / "demo_pinn" / "pinn.pt"

# Rebuild the same architecture used by examples/04_solve_pinn.py
model = PINNMLP({"hidden_dims": [32, 32, 32], "activation": "tanh"})
payload = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(payload["state_dict"])
model.eval()

Lx = Ly = 0.01
rng = np.random.default_rng(0)
# sample interior points away from the crack tip singularity for a fair comparison
n = 4000
x = rng.uniform(-Lx, Lx, n)
y = rng.uniform(-Ly, Ly, n)
# exclude points on the crack line near the tip (singular region)
r_tip = np.sqrt((x - CRACK_HALF_LENGTH) ** 2 + y ** 2)
mask = r_tip > 2e-4
pts = np.column_stack([x[mask], y[mask]]).astype(np.float32)

with torch.no_grad():
    u_pinn = model(torch.tensor(pts)).numpy()

ref = westergaard_i(pts, a=CRACK_HALF_LENGTH, sigma_inf=SIGMA_INF, E=E, nu=NU)

l2_x = l2_relative(u_pinn[:, 0], ref["ux"])
l2_y = l2_relative(u_pinn[:, 1], ref["uy"])

print("=== PINN vs Westergaard (Mode-I) ===")
print(f"evaluation points: {len(pts)}")
print(f"relative L2 ux: {l2_x*100:.2f} %")
print(f"relative L2 uy: {l2_y*100:.2f} %")
print(f"mean relative L2: {(l2_x+l2_y)/2*100:.2f} %")
print(f"PINN ux range: [{u_pinn[:,0].min():.3e}, {u_pinn[:,0].max():.3e}] m")
print(f"Wster ux range: [{ref['ux'].min():.3e}, {ref['ux'].max():.3e}] m")
