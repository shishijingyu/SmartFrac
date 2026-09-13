"""加速比统计（FR-2.11）。"""
from __future__ import annotations

import time

import numpy as np


def measure_step_time(run_fn, n_warmup: int = 5, n_measure: int = 20) -> float:
    """measure single-step wall time (s)."""
    for _ in range(n_warmup):
        run_fn()
    t0 = time.time()
    for _ in range(n_measure):
        run_fn()
    return (time.time() - t0) / n_measure


def speedup(neural_one_step: float, reference_one_step: float) -> float:
    return reference_one_step / max(neural_one_step, 1e-9)
