"""Глава 10. Геометрия в пространстве двух коэффициентов."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import matplotlib.pyplot as plt
from pure_core import lasso_cd, mse, predict
from plot_utils import savefig
from experiments_a import finish

X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
y = [2 * row[0] + .5 * row[1] for row in X]
fit = lasso_cd(X, y, alpha=.8, fit_intercept=False)
contact = fit["coef"]
radius = sum(abs(w) for w in contact)
contact_loss = mse(y, predict(X, contact)) / 2
assert max(abs(a - b) for a, b in zip(contact, [1.2, 0])) < 1e-8
RESULT = {"X": X, "y": y, "ols_coef": [2, .5],
          "lasso_coef": contact, "alpha": .8,
          "l1_radius": radius, "contact_half_mse": contact_loss,
          "objective": fit["objective"]}
FIGURES = ["f10_norms", "f10_lasso_contact"]

fig, ax = plt.subplots(figsize=(7.3, 5))
angles = [i * 2 * math.pi / 360 for i in range(361)]
ax.plot([math.cos(t) for t in angles], [math.sin(t) for t in angles],
        label="L2: w₁² + w₂² = 1")
ax.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0],
        label="L1: |w₁| + |w₂| = 1", color="#D55E00", lw=2)
ax.axhline(0, color="black", lw=.6)
ax.axvline(0, color="black", lw=.6)
ax.set(xlabel="Коэффициент w₁", ylabel="Коэффициент w₂",
       title="Оси показывают веса модели", aspect="equal",
       xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
ax.legend(loc="upper right", fontsize=11)
savefig(fig, FIGURES[0])

fig, ax = plt.subplots(figsize=(7.6, 5))
u = [-1.7 + i * .025 for i in range(177)]
v = [-1.7 + i * .025 for i in range(137)]
# Для этой X: MSE/2 = ((w1−2)^2 + (w2−.5)^2)/2.
surface = [[((a - 2) ** 2 + (b - .5) ** 2) / 2
            for a in u] for b in v]
lines = ax.contour(u, v, surface,
                    levels=[.1, contact_loss, .8, 1.4, 2.2],
                    colors="#0072B2", linewidths=1.1)
ax.clabel(lines, fmt="%.3g", fontsize=10)
ax.fill([radius, 0, -radius, 0], [0, radius, 0, -radius],
        alpha=.15, color="#D55E00")
ax.plot([radius, 0, -radius, 0, radius],
        [0, radius, 0, -radius, 0], color="#D55E00", lw=2,
        label="L1 ≤ 1.2")
ax.scatter([2], [.5], marker="x", color="black", s=80,
           label="OLS: (2; 0.5)")
ax.scatter([contact[0]], [contact[1]], s=65, c="#009E73",
           zorder=5, label="Lasso: (1.2; 0)")
ax.axhline(0, color="black", lw=.6)
ax.axvline(0, color="black", lw=.6)
ax.set(xlabel="Коэффициент w₁", ylabel="Коэффициент w₂",
       title="Линия равной ошибки касается вершины", aspect="equal")
ax.legend(loc="lower right", fontsize=11)
savefig(fig, FIGURES[1])
finish(10, RESULT)
