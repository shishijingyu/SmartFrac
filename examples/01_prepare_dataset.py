"""示例 1：M1 数据管线冒烟（30 分钟跑通）。"""
import numpy as np
from smartfrac.core.datatypes import AtomicSample
from smartfrac.io_utils.cleaning import filter_invalid
from smartfrac.io_utils.dedup import dedup
from smartfrac.io_utils.augment import augment
from smartfrac.io_utils.split import stratified_split

rng = np.random.default_rng(42)
samples = [AtomicSample(
    numbers=np.array([26, 26]),
    positions=rng.normal(size=(2, 3)),
    cell=np.eye(3),
    energy=float(rng.normal()),
    forces=rng.normal(size=(2, 3)),
) for _ in range(100)]

clean, log = filter_invalid(samples)
nodup, nd = dedup(clean)
aug = augment(nodup, n_aug=1)
energies = np.array([s.energy for s in aug])
sp = stratified_split(energies)
print(f"done: {len(aug)} samples, split={len(sp.train_idx)}/{len(sp.val_idx)}/{len(sp.test_idx)}")
