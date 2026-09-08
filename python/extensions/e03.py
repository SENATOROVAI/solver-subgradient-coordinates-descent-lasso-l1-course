"""Д03. Опорные прямые и один общий набор знаков."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from pure_core import mae, predict
from mae_extra import mae_gradient
from plot_utils import savefig
from extra_utils import finish_extra

FIGURES = ["e03_support_lines", "e03_shared_signs"]
grid = [k / 100 for k in range(-200, 201)]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.plot(grid, [abs(z) for z in grid], color="black", lw=2.5,
        label="|z|")
for g, style in [(-0.6, "-"), (0.0, "-"), (1.5, "--")]:
    ax.plot(grid, [g * z for z in grid], style, label=f"Наклон {g}")
ax.set(xlabel="z", ylabel="Функция и проверяемые прямые")
ax.legend()
savefig(fig, FIGURES[0])
X, y = [[1.0], [1.0], [0.5]], [0.0, 1.0, -1.0]
ws, bs = np.linspace(-1, 5, 181), np.linspace(-4, 1, 151)
W, B = np.meshgrid(ws, bs)
loss = np.zeros_like(W)
for row, target in zip(X, y):
    loss += np.abs(target - row[0] * W - B) / len(y)
fig, ax = plt.subplots(figsize=(7.6, 4.7))
cs = ax.contour(W, B, loss, colors="#24577C",
                levels=[1/3 + 0.002, .45, .6, .8, 1, 1.5])
ax.clabel(cs, fmt="%.2f", fontsize=11)
for x, target in [(1, 0), (1, 1), (.5, -1)]:
    ax.plot(ws, target - x * ws, ":", color="#8E598C", alpha=.6)
ax.scatter([0, .6, 2], [0, -.6, -2], s=65, zorder=4)
ax.annotate("Пробный совместный шаг", (.6, -.6), (1.3, .4),
            arrowprops={"arrowstyle": "->"})
ax.annotate("Минимум", (2, -2), (3.7, -2.8),
            arrowprops={"arrowstyle": "->"}, bbox={"fc": "w", "ec": "w"})
ax.set(xlabel="Коэффициент w", ylabel="Свободный член b",
       xlim=(-1, 5), ylim=(-4, 1))
savefig(fig, FIGURES[1])
grad_w, grad_b = mae_gradient([[0], [0], [0]], [0, 1, 1], [0], 1)
initial = mae(y, predict(X, [0], 0))
shifted = mae(y, predict(X, [.6], -.6))
assert grad_b == 1 / 3
assert shifted < initial
for g in [-1, -.6, 0, .5, 1]:
    assert all(abs(z) + 1e-12 >= g * z for z in grid)
assert abs(2) < 1.5 * 2
RESULT = {"median_sign_gradient": grad_b,
          "median_full_subdifferential": [-1 / 3, 1],
          "initial_mae": initial, "joint_step_mae": shifted,
          "known_minimum": mae(y, predict(X, [2], -2)),
          "zero_residual_sign_needed_for_b": 0,
          "zero_residual_sign_needed_for_w": -.5,
          "support_valid": [-1, -.6, 0, .5, 1],
          "invalid_support": 1.5}
finish_extra("e03", RESULT)
