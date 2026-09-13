"""单元测试：NNP / PINN 前向与 PDE。"""
import numpy as np
import torch

from smartfrac.nnp.model import ElementEnergyNet
from smartfrac.nnp.loss import nnp_loss
from smartfrac.pinn.model import PINNMLP
from smartfrac.pinn.benchmarks.westergaard import westergaard_i, l2_relative


def test_nnp_forward_shape():
    m = ElementEnergyNet({"descriptor_dim": 50, "hidden_dims": [32, 32],
                          "species": [26]})
    desc = torch.randn(10, 50)
    species = torch.full((10,), 26, dtype=torch.long)
    out = m(desc, species)
    assert out.shape == (10,)


def test_nnp_loss_finite():
    pe = torch.randn(10); te = torch.randn(10)
    pf = torch.randn(10); tf = torch.randn(10)
    r = nnp_loss(pe, te, pf, tf)
    assert torch.isfinite(r["loss"])


def test_pinn_forward():
    m = PINNMLP({"hidden_dims": [16, 16]})
    xy = torch.randn(20, 2)
    out = m(xy)
    assert out.shape == (20, 2)


def test_westergaard_shapes():
    xy = np.random.rand(50, 2) * 0.002
    r = westergaard_i(xy, a=0.005, sigma_inf=2e8, E=2.1e11, nu=0.3)
    assert r["ux"].shape == (50,)
    assert np.isfinite(r["ux"]).all()
