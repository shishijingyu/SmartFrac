"""示例 4：PINN 单裂纹求解最小 demo。"""
from smartfrac.constants import CRACK_HALF_LENGTH
from smartfrac.pinn.model import PINNMLP
from smartfrac.pinn.trainer import PINNTrainer
from smartfrac.pinn.sampling import sample_domain, sample_boundary
from smartfrac.viz.curves import plot_loss_curve

model = PINNMLP({"hidden_dims": [32, 32, 32], "activation": "tanh"})
trainer = PINNTrainer(model, {"epochs": 200, "lr": 1e-3,
                              "n_collocation": 2000, "n_boundary": 200,
                              "Lx": 0.01, "Ly": 0.01, "tip_radius": 0.0005},
                      out_dir="results/demo_pinn")
domain = sample_domain(2000, 0.01, 0.01, (CRACK_HALF_LENGTH, 0.0), tip_radius=0.0005)
bc = sample_boundary(200, 0.01, 0.01)
hist = trainer.fit(domain, bc)

# 画 loss 曲线
plot_loss_curve(hist, out="results/demo_pinn/loss.png")
print(f"PINN demo done -> results/demo_pinn  (loss: {hist['loss'][0]:.2e} -> {hist['loss'][-1]:.2e})")
