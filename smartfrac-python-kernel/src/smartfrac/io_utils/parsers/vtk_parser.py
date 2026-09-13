"""VTK/VTU 场数据解析（FR-1.3，PINN/FEM 基准用）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from smartfrac.core.registry import PARSER_REGISTRY
from smartfrac.core.datatypes import FieldData
from .base import DataParser


class VTKFieldParser(DataParser):
    """读取 .vtu/.vtk 网格点云，输出 FieldData。"""

    def parse(self, path: Path) -> list[AtomicSample]:  # type: ignore[override]
        raise NotImplementedError("场数据用 load_field()")

    def load_field(self, path: Path) -> FieldData:
        mesh = pv.read(str(path))
        coords = np.asarray(mesh.points)[:, :2]
        def _get(name: str) -> np.ndarray:
            return np.asarray(mesh.point_data[name]) if name in mesh.point_data else np.zeros(len(coords))
        return FieldData(
            coords=coords,
            ux=_get("ux"), uy=_get("uy"),
            sxx=_get("sxx"), syy=_get("syy"), sxy=_get("sxy"),
            meta={"source": str(path)},
        )


@PARSER_REGISTRY.register(".vtu")
class VTUParser(VTKFieldParser):
    ext = ".vtu"


@PARSER_REGISTRY.register(".vtk")
class VTKParser(VTKFieldParser):
    ext = ".vtk"
