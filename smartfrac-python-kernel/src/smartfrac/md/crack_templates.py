"""裂纹初始构型模板（FR-2.8）：中心裂纹 CC / 单边缺口 SEB。"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk

from ..constants import ALPHA_FE


def make_cc_2d(size=(10, 10, 1), crack_length: float = 4.0,
               element: str = "Fe") -> Atoms:
    """中心裂纹板（二维，沿 z 压薄）。在 x=0 附近删原子形成裂纹。"""
    at = bulk(element, "bcc", a=ALPHA_FE["lattice_a_A"], cubic=True)
    at = at.repeat(size)
    pos = at.get_positions()
    # 删除裂纹区间：|y| < crack_length/2 且 |x| < 0.5
    mask = ~((np.abs(pos[:, 1]) < crack_length / 2) & (np.abs(pos[:, 0]) < 0.6))
    del at[np.where(~mask)[0]]
    at.info["crack_type"] = "CC-2D"
    at.info["crack_length"] = crack_length
    return at


def make_seb_3d(size=(8, 8, 4), notch_length: float = 3.0,
                element: str = "Fe") -> Atoms:
    """单边缺口梁（三维）。"""
    at = bulk(element, "bcc", a=ALPHA_FE["lattice_a_A"], cubic=True)
    at = at.repeat(size)
    pos = at.get_positions()
    xmax = pos[:, 0].max()
    mask = ~((pos[:, 0] > xmax - notch_length) & (np.abs(pos[:, 1]) < 1.5))
    del at[np.where(~mask)[0]]
    at.info["crack_type"] = "SEB-3D"
    at.info["notch_length"] = notch_length
    return at
