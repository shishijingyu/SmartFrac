"""Dimensionless PINN: rescale coordinates to [-1,1] and displacements by the
remote-field displacement scale delta = sigma_inf * L / E, so that the network
operates on O(1) quantities and the PDE / BC losses are commensurate.

This isolates the *physics* (linear elasticity equilibrium) from the
*engineering-unit* scaling, which is the standard PINN conditioning step.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from smartfrac.constants import E, NU, SIGMA_INF, CRACK_HALF_LENGTH

L = 0.01                      # half side length (m)
delta = SIGMA_INF * L / E     # reference displacement scale (m)
a_norm = CRACK_HALF_LENGTH / L   # dimensionless crack half-length = 0.5


def autograd(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def residual(xyn, model, nu):
    """Dimensionless plane-stress residual on x,y in [-1,1]."""
    xyn = xyn.clone().requires_grad_(True)
    u = model(xyn)
    ux, uy = u[:, 0:1], u[:, 1:2]
    dux = autograd(ux, xyn)
    duy = autograd(uy, xyn)
    exx, exy = dux[:, 0:1], 0.5 * (dux[:, 1:2] + duy[:, 0:1])
    eyy = duy[:, 1:2]
    c = 1.0 / (1 - nu ** 2)
    sxx = c * (exx + nu * eyy)
    syy = c * (eyy + nu * exx)
    sxy = c * (1 - nu) * exy
    rx = autograd(sxx, xyn)[:, 0:1] + autograd(sxy, xyn)[:, 1:2]
    ry = autograd(sxy, xyn)[:, 0:1] + autograd(syy, xyn)[:, 1:2]
    return rx, ry


class MLP(nn.Module):
    def __init__(self, w=(64, 64, 64, 64)):
        super().__init__()
        layers, prev = [], 2
        for h in w:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers += [nn.Linear(prev, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


torch.manual_seed(0)
model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

rng = np.random.default_rng(0)
n_col, n_bc = 8000, 1000

# interior collocation, tip-refined near (a_norm, 0) and (-a_norm, 0)
uni = np.column_stack([rng.uniform(-1, 1, n_col - 1600),
                       rng.uniform(-1, 1, n_col - 1600)])
def tip_pts(center, n):
    r = 0.25 * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([center[0] + r * np.cos(th), center[1] + r * np.sin(th)])
tip = np.vstack([tip_pts((a_norm, 0), 800), tip_pts((-a_norm, 0), 800)])
col = torch.tensor(np.vstack([uni, tip]), dtype=torch.float32)

# boundary points + far-field equi-biaxial Dirichlet (matches the standard
# Westergaard Mode-I form: Ux = xi, Uy = eta)
def bc_pts(n):
    pts, side = [], rng.integers(0, 4, n)
    for s in side:
        if s == 0: pts.append([-1.0, rng.uniform(-1, 1)])
        elif s == 1: pts.append([1.0, rng.uniform(-1, 1)])
        elif s == 2: pts.append([rng.uniform(-1, 1), -1.0])
        else: pts.append([rng.uniform(-1, 1), 1.0])
    return torch.tensor(pts, dtype=torch.float32)
bc = bc_pts(n_bc)
bc_ref = torch.stack([(1 - NU) * bc[:, 0], (1 - NU) * bc[:, 1]], dim=1)

loss_hist = []
for ep in range(1, 3001):
    opt.zero_grad()
    rx, ry = residual(col, model, NU)
    pde = (rx ** 2 + ry ** 2).mean()
    bcp = model(bc)
    bc_l = ((bcp - bc_ref) ** 2).mean()
    loss = pde + 10.0 * bc_l
    loss.backward()
    opt.step()
    loss_hist.append(float(loss))
    if ep % 500 == 0:
        print(f"ep {ep}: loss={loss:.4e} pde={pde:.3e} bc={bc_l:.3e}")

# ---- evaluate against correct Westergaard in dimensionless units ----
def westergaard_dimless(pts, a, nu):
    x, y = pts[:, 0], pts[:, 1]
    z = x + 1j * y
    Z = z / np.sqrt(z ** 2 - a ** 2 + 0j)
    Zp = -a ** 2 / (z ** 2 - a ** 2 + 0j) ** 1.5
    Zbar = np.sqrt(z ** 2 - a ** 2 + 0j)
    ux = (1 - nu) * Zbar.real - (1 + nu) * y * Z.imag
    uy = (1 - nu) * Zbar.imag + (1 + nu) * y * Z.real
    return ux, uy

rng2 = np.random.default_rng(9)
m = 6000
xv = rng2.uniform(-1, 1, m)
yv = rng2.uniform(-1, 1, m)
r1 = np.sqrt((xv - a_norm) ** 2 + yv ** 2)
r2 = np.sqrt((xv + a_norm) ** 2 + yv ** 2)
keep = (r1 > 0.04) & (r2 > 0.04)
pts = np.column_stack([xv[keep], yv[keep]]).astype(np.float32)

model.eval()
with torch.no_grad():
    up = model(torch.tensor(pts)).numpy()
ur, vr = westergaard_dimless(pts, a_norm, NU)

def l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))

lx, ly = l2(up[:, 0], ur), l2(up[:, 1], vr)
print("=== Dimensionless PINN vs Westergaard (far-field BC only) ===")
print(f"points: {len(pts)}")
print(f"relative L2 ux: {lx*100:.2f} %")
print(f"relative L2 uy: {ly*100:.2f} %")
print(f"mean L2: {(lx+ly)/2*100:.2f} %")
# also: error in the far field (|x|>0.8) vs near crack
far = (np.abs(pts[:, 0]) > 0.8) | (np.abs(pts[:, 1]) > 0.8)
print(f"far-field mean L2: {(l2(up[far,0],ur[far])+l2(up[far,1],vr[far]))/2*100:.2f} %")
