"""SOAP 描述符封装（FR-2.1，DScribe）。"""
from __future__ import annotations

import numpy as np
from dscribe.descriptors import SOAP
from ase import Atoms


class SoapDescriptor:
    """统一封装 DScribe SOAP。"""

    def __init__(self, species: list[int], r_cut: float = 5.0,
                 n_max: int = 8, l_max: int = 6):
        self.species = sorted(species)
        self.soap = SOAP(species=self.species, r_cut=r_cut, n_max=n_max,
                         l_max=l_max, periodic=False, average="off")
        self.dim = self.soap.get_number_of_features()

    def transform(self, atoms_list: list[Atoms]) -> np.ndarray:
        """返回 (M*N_atoms, D) 的每原子描述符（拼平）。"""
        feats = self.soap.create(atoms_list, n_jobs=1)
        return np.vstack(feats) if isinstance(feats, list) else np.asarray(feats)
