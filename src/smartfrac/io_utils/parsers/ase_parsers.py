"""ASE 可读格式统一解析（xyz/POSCAR/cif/cube/vasprun）。"""
from __future__ import annotations

from pathlib import Path

from ase import io as ase_io

from smartfrac.core.datatypes import AtomicSample
from smartfrac.core.registry import PARSER_REGISTRY
from .base import DataParser


@PARSER_REGISTRY.register(".xyz")
class XYZParser(DataParser):
    ext = ".xyz"

    def parse(self, path: Path) -> list[AtomicSample]:
        return _read_ase(path)


@PARSER_REGISTRY.register(".poscar")
class PoscarParser(DataParser):
    ext = ".poscar"

    def parse(self, path: Path) -> list[AtomicSample]:
        return _read_ase(path)


@PARSER_REGISTRY.register(".cif")
class CifParser(DataParser):
    ext = ".cif"

    def parse(self, path: Path) -> list[AtomicSample]:
        return _read_ase(path)


@PARSER_REGISTRY.register(".cube")
class CubeParser(DataParser):
    ext = ".cube"

    def parse(self, path: Path) -> list[AtomicSample]:
        return _read_ase(path)


@PARSER_REGISTRY.register(".vasp")
class VaspParser(DataParser):
    ext = ".vasp"

    def parse(self, path: Path) -> list[AtomicSample]:
        return _read_ase(path)


def _read_ase(path: Path) -> list[AtomicSample]:
    frames = ase_io.read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    out: list[AtomicSample] = []
    for at in frames:
        out.append(AtomicSample(
            numbers=at.get_atomic_numbers(),
            positions=at.get_positions(),
            cell=at.get_cell().array,
            energy=at.get_potential_energy() if "energy" in at.info else None,
            forces=at.get_forces() if "forces" in at.arrays else None,
        ))
    return out
