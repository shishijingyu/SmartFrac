"""全局物理常数与单位制（NFR-6：禁止散落硬编码）。

单位约定：
- 长度：Å
- 能量：eV
- 力：eV/Å
- 应力：GPa（连续介质 PINN 部分）
"""
from __future__ import annotations

# --- 转换常数 ---
EV_TO_J = 1.602176634e-19        # 1 eV = J
ANGSTROM_TO_M = 1e-10            # 1 Å = m
PA_TO_GPA = 1e-9                  # 1 Pa = 1e-9 GPa

# --- α-Fe (BCC) 基准材料（需求 6.2）---
ALPHA_FE = {
    "E_GPa": 210.0,               # 杨氏模量
    "nu": 0.3,                     # 泊松比
    "sigma_inf_MPa": 200.0,        # 远场拉应力
    "crack_half_length_mm": 5.0,  # a, 中心裂纹长度 2a=10mm
    "lattice_a_A": 2.8553,         # BCC 晶格常数
}

# 连续介质换算：GPa（PINN 内部用无量纲/国际单位自洽）
E = ALPHA_FE["E_GPa"] * 1e9        # Pa
NU = ALPHA_FE["nu"]
SIGMA_INF = ALPHA_FE["sigma_inf_MPa"] * 1e6  # Pa
CRACK_HALF_LENGTH = ALPHA_FE["crack_half_length_mm"] * 1e-3  # m
