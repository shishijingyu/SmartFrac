"""重复构型去重（FR-1.8）：基于粗粒度特征哈希。"""
from __future__ import annotations

import hashlib

import numpy as np

from smartfrac.core.datatypes import AtomicSample


def _fingerprint(s: AtomicSample) -> str:
    """用排序后原子坐标的低精度量化做指纹。"""
    hist, _ = np.histogram(s.positions, bins=20, range=(-5, 10))
    quant = np.round(s.positions, 1).tobytes()
    return hashlib.md5(hist.tobytes() + quant).hexdigest()


def dedup(samples: list[AtomicSample]) -> tuple[list[AtomicSample], int]:
    seen: set[str] = set()
    out: list[AtomicSample] = []
    dropped = 0
    for s in samples:
        fp = _fingerprint(s)
        if fp in seen:
            dropped += 1
            continue
        seen.add(fp)
        out.append(s)
    return out, dropped
