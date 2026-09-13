"""PINN 包。"""
from .model import PINNMLP  # noqa: F401
from .pde import elastic_residual  # noqa: F401
from .trainer import PINNTrainer  # noqa: F401
from . import benchmarks  # noqa: F401
