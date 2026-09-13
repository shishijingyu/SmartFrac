"""LAMMPS dump 文件解析（FR-1.1）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from smartfrac.core.datatypes import AtomicSample
from smartfrac.core.registry import PARSER_REGISTRY
from .base import DataParser


@PARSER_REGISTRY.register(".dump")
class LammpsDumpParser(DataParser):
    ext = ".dump"

    def parse(self, path: Path) -> list[AtomicSample]:
        samples: list[AtomicSample] = []
        natoms = 0
        cell = np.eye(3)
        data: list[list[float]] = []
        cols: list[str] = []

        def flush() -> None:
            nonlocal data, cols
            if not data:
                return
            arr = np.array(data)
            idx = {c: i for i, c in enumerate(cols)}
            ids = arr[:, idx["id"]].astype(int) if "id" in idx else np.arange(len(arr))
            order = np.argsort(ids)
            arr = arr[order]
            positions = np.column_stack([arr[:, idx["x"]], arr[:, idx["y"]],
                                         arr[:, idx.get("z", np.zeros(len(arr)))]]
                                        if "z" in idx else
                                        [arr[:, idx["x"]], arr[:, idx["y"]], np.zeros(len(arr))])
            types = arr[:, idx["type"]].astype(int)
            # 元素序号占位：dump 里只有 type id，调用方需在 metadata 提供 type->Z 映射
            samples.append(AtomicSample(
                numbers=types, positions=positions, cell=cell))
            data = []

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s.startswith("ITEM: NUMBER OF ATOMS"):
                    flush()
                    natoms = int(next(f).split()[0])
                elif s.startswith("ITEM: BOX BOUNDS"):
                    xs, ys, zs = [], [], []
                    for _ in range(3):
                        lo, hi = map(float, next(f).split()[:2])
                        xs.append(lo); ys.append(lo); zs.append(lo)
                    cell = np.diag([xs[-1] - xs[0], 0, 0])
                    # 简化：仅取对角盒子
                    box = np.zeros((3, 3))
                    for i, (lo, hi) in enumerate(
                            [tuple(map(float, next(f).split()[:2])) for _ in range(0)]):
                        pass
                elif s.startswith("ITEM: ATOMS"):
                    cols = s.split()[2:]
                elif s.startswith("ITEM: TIMESTEP") or s.startswith("ITEM:"):
                    continue
                elif s and s[0].isdigit():
                    data.append([float(x) for x in s.split()])
            flush()
        return samples
