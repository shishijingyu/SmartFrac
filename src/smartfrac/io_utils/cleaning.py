"""数据清洗（FR-1.5）：剔除 NaN/Inf，输出清洗日志。"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from smartfrac.core.datatypes import AtomicSample

log = logging.getLogger(__name__)


def filter_invalid(samples: Iterable[AtomicSample]) -> tuple[list[AtomicSample], dict]:
    """剔除坐标/能量/力含 NaN 或 Inf 的构型。返回(干净样本, 日志)。"""
    kept, dropped = [], {"nan_coord": 0, "nan_energy": 0, "nan_force": 0, "total": 0}
    for s in samples:
        dropped["total"] += 1
        bad = False
        if not np.all(np.isfinite(s.positions)):
            dropped["nan_coord"] += 1
            bad = True
        if s.energy is not None and not np.isfinite(s.energy):
            dropped["nan_energy"] += 1
            bad = True
        if s.forces is not None and not np.all(np.isfinite(s.forces)):
            dropped["nan_force"] += 1
            bad = True
        if not bad:
            kept.append(s)
    log.info(f"clean: {len(kept)}/{dropped['total']} kept, dropped={dropped}")
    return kept, dropped
