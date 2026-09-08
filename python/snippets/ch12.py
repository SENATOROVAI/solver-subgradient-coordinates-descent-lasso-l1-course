"""Глава 12: воспроизводимый эксперимент."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from plot_utils import savefig
from pure_core import lasso_cd
from experiments_b import (
    finish, pair, mark_axes, report_fit, COLORS)


def ch12():
    X = [[-2., -1.], [-1., -1.5], [0., -.5],
         [0., .5], [1., 1.5], [2., 1.]]
    y = [-5., -3.5, -.5, .5, 3.5, 5.]
    alpha = .2
    fit = lasso_cd(X, y, alpha=alpha, fit_intercept=False,
                   keep_steps=True, tol=1e-10)
    trace = fit["trace"]
    matrix = np.array(X)
    target = np.array(y)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    frame_records = []
    for index, ax in enumerate(axes.flat, start=1):
        previous = np.array(trace[index - 1]["coef"])
        current = np.array(trace[index]["coef"])
        j = trace[index]["coordinate"]
        partial = target - matrix @ previous + matrix[:, j] * previous[j]
        q = float(np.mean(matrix[:, j] ** 2))
        z = float(np.mean(matrix[:, j] * partial))
        xx = np.linspace(matrix[:, j].min(), matrix[:, j].max(), 50)
        ax.scatter(matrix[:, j], partial, color=COLORS[0], s=30,
                   label="Частичный остаток")
        ax.plot(xx, previous[j] * xx, "--", color=".55",
                label="До обновления")
        ax.plot(xx, current[j] * xx, color=COLORS[1],
                label="После порога")
        ax.set_title(f"Шаг {index}: w{j + 1} = {current[j]:.3f}")
        mark_axes(ax, f"Признак x{j + 1}", "y − вклад другого признака")
        ax.text(.03, .96, f"q = {q:.3f}; z = {z:.3f}",
                transform=ax.transAxes, va="top", fontsize=11)
        if index == 1:
            ax.legend(loc="lower right", fontsize=11)
        frame_records.append({"coordinate": j, "q": q, "z": z,
                              "before": previous.tolist(),
                              "after": current.tolist(),
                              "partial_residual": partial.tolist()})
    savefig(fig, "f12_coordinate_frames")
    fig, axes = pair()
    u = np.linspace(-.3, 2.8, 220)
    v = np.linspace(-.8, 1.2, 220)
    U, V = np.meshgrid(u, v)
    loss = np.zeros_like(U)
    for row, value in zip(X, y):
        loss += (value - row[0] * U - row[1] * V) ** 2 / (2 * len(y))
    loss += alpha * (np.abs(U) + np.abs(V))
    levels = [fit["objective"] + d for d in [.02, .1, .3, .8, 2, 4]]
    axes[0].contour(U, V, loss, levels=levels, colors=".6")
    states = np.array([row["coef"] for row in trace])
    axes[0].plot(states[:, 0], states[:, 1], "o-", color=COLORS[1],
                 markersize=3, label="Каждое обновление")
    axes[0].scatter(*states[-1], s=70, color=COLORS[0], zorder=5)
    mark_axes(axes[0], "Коэффициент w1", "Коэффициент w2")
    axes[0].legend(fontsize=11)
    scores = [row["objective"] for row in trace]
    axes[1].plot(range(len(scores)), scores, color=COLORS[0])
    mark_axes(axes[1], "Число обновлений координат", "Полный критерий J")
    savefig(fig, "f12_cd_path")
    assert all(a + 1e-10 >= b for a, b in zip(scores, scores[1:]))
    return finish(12, {"X": X, "y": y, "alpha": alpha,
                       "frames": frame_records, "coef": fit["coef"],
                       "objective": fit["objective"],
                       "diagnostic": report_fit(fit)})


RESULT = ch12()
FIGURES = ['f12_coordinate_frames', 'f12_cd_path']
