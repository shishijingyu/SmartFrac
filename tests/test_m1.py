"""单元测试：M1 清洗/归一化/划分。"""
import numpy as np
import pytest

from smartfrac.core.datatypes import AtomicSample
from smartfrac.io_utils.cleaning import filter_invalid
from smartfrac.io_utils.normalization import Normalizer
from smartfrac.io_utils.split import stratified_split


def _sample(n=10):
    rng = np.random.default_rng(0)
    return [AtomicSample(numbers=np.array([26, 26]),
                         positions=rng.normal(size=(2, 3)),
                         cell=np.eye(3),
                         energy=float(rng.normal()),
                         forces=rng.normal(size=(2, 3))) for _ in range(n)]


def test_filter_invalid_drops_nan():
    s = _sample(5)
    s[2].positions[0, 0] = np.nan
    kept, log = filter_invalid(s)
    assert len(kept) == 4
    assert log["nan_coord"] == 1


def test_normalizer_roundtrip():
    n = Normalizer()
    e = np.array([1.0, 2.0, 3.0, 4.0])
    f = np.random.default_rng(0).normal(size=(10, 3))
    n.fit_labels(e, f)
    e_t = n.transform_energy(e)
    e_back = n.inverse_energy(e_t)
    assert np.allclose(e, e_back, atol=1e-6)


def test_stratified_split_sizes():
    e = np.arange(100, dtype=float)
    sp = stratified_split(e)
    assert len(sp.train_idx) + len(sp.val_idx) + len(sp.test_idx) == 100
