"""Глава 05. Сетка и фактические шаги субградиентного метода."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from pure_core import grid_mae, fit_mae, mae, predict
from plot_utils import savefig
from experiments_a import finish

X, y = [[1], [2], [3]], [3, 5, 8]
ws = [1 + i * .025 for i in range(81)]
bs = [-1 + i * .025 for i in range(161)]
grid = grid_mae(X, y, ws, bs)
good = fit_mae(X, y, steps=2500, step=.12, decay=.6,
               w0=[0], b0=0)
bad = fit_mae(X, y, steps=2500, step=1., decay=0,
              w0=[0], b0=0)
def summary(fit):
    return {k: fit[k] for k in ["coef", "intercept", "best_loss",
            "n_iter", "status", "converged"]} | {
                "last_loss": fit["history"][-1],
                "last_parameters": fit["parameter_history"][-1]}
RESULT = {"grid": {k: grid[k] for k in ["coef", "intercept", "loss"]},
          "good": summary(good), "constant": summary(bad),
          "good_first_losses": good["history"][:12],
          "constant_first_losses": bad["history"][:12],
          "settings": {"steps": 2500, "good_step": .12,
                       "good_decay": .6, "constant_step": 1.}}
FIGURES = ["f05_grid", "f05_trajectories", "f05_steps"]
# Матрица только для рисования: расчеты остаются на списках.
loss_map = [[mae(y, predict(X, [w], b)) for w in ws] for b in bs]
fig, ax = plt.subplots(figsize=(8, 4.7))
mesh = ax.pcolormesh(ws, bs, loss_map, shading="auto", cmap="viridis")
fig.colorbar(mesh, ax=ax, label="MAE")
ax.scatter(grid["coef"], [grid["intercept"]], marker="*", s=180,
           c="white", edgecolors="black", label="Минимум этой сетки")
ax.set(xlabel="Коэффициент w", ylabel="Свободный член b",
       title="Разные пары параметров дают разную ошибку")
ax.legend(fontsize=11)
savefig(fig, FIGURES[0])

fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
for ax, fit, title in [(axes[0], good, "Убывающий шаг"),
                       (axes[1], bad, "Постоянный большой шаг")]:
    trace = fit["parameter_history"]
    ax.plot([p[0] for p in trace], [p[1] for p in trace], lw=.9)
    ax.scatter([trace[0][0]], [trace[0][1]], c="black", label="Старт")
    ax.scatter(fit["coef"], [fit["intercept"]], marker="*", s=140,
               color="#D55E00", label="Лучший посещенный")
    ax.scatter(grid["coef"], [grid["intercept"]], marker="x", s=70,
               color="#009E73", label="Минимум сетки")
    ax.set(xlabel="w", ylabel="b", title=title)
    ax.legend(fontsize=11)
savefig(fig, FIGURES[1])

fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
for fit, label in [(good, "Убывающий"), (bad, "Постоянный")]:
    axes[0].plot(range(81), fit["history"][:81], label=label)
axes[1].plot(range(2400, 2501), good["history"][2400:],
             label="Убывающий", lw=1.2)
titles = ["Первые 80 шагов: оба метода", "Убывающий: последние 100 шагов"]
for ax, title in zip(axes, titles):
    ax.axhline(grid["loss"], ls="--", color="black", label="Сетка")
    ax.set(xlabel="Номер шага", ylabel="MAE текущей точки", title=title)
    ax.legend(fontsize=11)
savefig(fig, FIGURES[2])
finish(5, RESULT)
