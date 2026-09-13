"""示例 5：可视化冒烟——用随机数据画 4 类图（TC-07）。"""
from pathlib import Path

import numpy as np

from smartfrac.viz.curves import parity_plot, plot_profile, plot_loss_curve
from smartfrac.viz.field import plot_field_contour

out = Path("results/demo_figures")
out.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)

# 1) parity plot（预测 vs 参考）
ref = rng.normal(size=200)
pred = ref + 0.1 * rng.normal(size=200)
parity_plot(pred, ref, out=str(out / "parity.png"))

# 2) 剖面对比曲线
x = np.linspace(0, 1, 100)
y_ref = np.sin(x)
y_pred = np.sin(x) + 0.05 * rng.normal(size=100)
plot_profile(x, y_ref, y_pred, label="ux", out=str(out / "profile.png"))

# 3) 场云图
n = 500
coords = rng.uniform(-1, 1, (n, 2))
values = np.exp(-(coords[:, 0] ** 2 + coords[:, 1] ** 2) / 0.1)
plot_field_contour(coords, values, title="sigma_xx", out=str(out / "field.png"))

# 4) loss 曲线（从 PINN demo 读取或造一条）
fake_hist = {"loss": list(np.logspace(20, 15, 200) + rng.normal(0, 1e14, 200))}
plot_loss_curve(fake_hist, out=str(out / "loss_demo.png"))

print(f"4 张图已生成到 {out.resolve()}:")
for p in sorted(out.glob("*.png")):
    print(" -", p.name, f"({p.stat().st_size//1024} KB)")
