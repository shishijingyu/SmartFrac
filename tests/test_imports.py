"""Smoke tests: verify the first-stage MVP toolchain imports correctly."""
from __future__ import annotations

import importlib

import pytest


REQUIRED_MODULES = [
    "numpy",
    "scipy",
    "pandas",
    "h5py",
    "yaml",
    "omegaconf",
    "torch",
    "ase",
    "dscribe",
    "matplotlib",
    "pyvista",
    "seaborn",
    "sklearn",
    "tensorboard",
]


@pytest.mark.parametrize("mod", REQUIRED_MODULES)
def test_import_required_modules(mod):
    """Every first-stage dependency must be importable."""
    imported = importlib.import_module(mod)
    assert imported is not None


def test_pytorch_version():
    import torch

    major, minor = torch.__version__.split(".")[:2]
    assert (int(major), int(minor)) >= (2, 0), f"PyTorch >= 2.0 required, got {torch.__version__}"


def test_config_loads():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/default.yaml")
    assert cfg.project.name == "SmartFrac"
    assert cfg.nnp.descriptor.name == "soap"
