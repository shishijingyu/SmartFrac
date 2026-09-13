"""FEM 基准加载（FR-3.8）。"""
from __future__ import annotations

from pathlib import Path

from smartfrac.io_utils.parsers.vtk_parser import VTKFieldParser


def load_fem_field(path: str | Path):
    return VTKFieldParser().load_field(Path(path))
