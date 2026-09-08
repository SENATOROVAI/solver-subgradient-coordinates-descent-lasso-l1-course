"""Глава 15: воспроизводимый эксперимент."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from plot_utils import savefig
from pure_core import coordinate_descent
from experiments_b import (
    finish, mark_axes, orthogonal_data, COLORS)


def ch15():
    X, y = orthogonal_data()
    lam = .5
    fit = coordinate_descent(X, y, alpha=lam, l1_ratio=0.,
                             fit_intercept=False, tol=1e-12)
    ref = Ridge(alpha=len(y) * lam, fit_intercept=False).fit(X, y)
    point = np.array(fit["coef"])
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    u = np.linspace(-1.7, 2.6, 300)
    v = np.linspace(-1.7, 2.1, 300)
    U, V = np.meshgrid(u, v)
    loss = .5 * ((U - 2) ** 2 + (V - .5) ** 2)
    touching_loss = .5 * np.sum((point - [2., .5]) ** 2)
    ax.contour(U, V, loss, levels=[touching_loss, .6, 1.2, 2.2],
               colors=[COLORS[1], ".7", ".7", ".7"])
    theta = np.linspace(0, 2 * np.pi, 600)
    radius = np.linalg.norm(point)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta),
            color=COLORS[0], lw=2, label="Окружность через решение Ridge")
    ax.scatter(*point, color=COLORS[0], s=70, zorder=4,
               label="Ridge: (4/3; 1/3)")
    ax.scatter(2, .5, marker="x", color=COLORS[1], s=70,
               label="Без штрафа: (2; 0,5)")
    ax.axhline(0, color=".4", lw=.7)
    ax.axvline(0, color=".4", lw=.7)
    ax.set_aspect("equal")
    mark_axes(ax, "Коэффициент w1", "Коэффициент w2")
    ax.legend(fontsize=11, loc="lower right")
    savefig(fig, "f15_ridge_geometry")
    strengths = np.linspace(0, 4, 80)
    weights = []
    errors = []
    for value in strengths:
        run = coordinate_descent(X, y, alpha=float(value),
                                 l1_ratio=0., fit_intercept=False)
        library = Ridge(alpha=len(y) * value,
                        fit_intercept=False).fit(X, y)
        weights.append(run["coef"])
        errors.append(np.max(np.abs(run["coef"] - library.coef_)))
    weights = np.array(weights)
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for j in range(2):
        ax.plot(strengths, weights[:, j], color=COLORS[j],
                lw=2, label=f"w{j + 1}")
    ax.axhline(0, color=".4", lw=.7)
    mark_axes(ax, "Сила квадратичного штрафа λ₂", "Коэффициент")
    ax.legend()
    savefig(fig, "f15_ridge_weights")
    error = float(np.max(np.abs(point - ref.coef_)))
    assert error < 1e-10 and max(errors) < 1e-8
    return finish(15, {"X": X, "y": y, "lambda2": lam,
                       "alpha_R": len(y) * lam, "coef": fit["coef"],
                       "sklearn_coef": ref.coef_.tolist(),
                       "max_path_error": float(max(errors)),
                       "touching_loss": float(touching_loss),
                       "constraint_radius": float(radius)})


RESULT = ch15()
FIGURES = ['f15_ridge_geometry', 'f15_ridge_weights']
