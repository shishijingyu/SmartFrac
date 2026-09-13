"""NNP ASE Calculator（FR-2.7）：PyTorch 模型包装成 ASE calculator。"""
from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes


class NNPCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, descriptor, device: str = "cpu"):
        super().__init__()
        self.model = model.to(device).eval()
        self.desc = descriptor
        self.device = device

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        from ase import Atoms
        atoms = atoms or self.atoms
        # 描述符
        feat = self.desc.soap.create([atoms], n_jobs=1)[0]
        desc = torch.tensor(feat, dtype=torch.float32, device=self.device)
        species = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long,
                               device=self.device)
        # 原子位置梯度（力）
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32,
                            device=self.device, requires_grad=True)
        with torch.enable_grad():
            per_atom = self.model(desc, species)
            e = per_atom.sum()
            f = -torch.autograd.grad(e, pos)[0]
        self.results = {"energy": float(e.item()), "forces": f.detach().cpu().numpy()}
