"""Глава 16: воспроизводимый эксперимент."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from plot_utils import savefig
from pure_core import elasticnet_cd
from experiments_b import (
    finish, pair, mark_axes, COLORS)


def ch16():
    grid = np.linspace(-3., 3., 601)
    U, V = np.meshgrid(grid, grid)
    l1 = .2 * (np.abs(U) + np.abs(V))
    en = l1 + .8 * (U ** 2 + V ** 2) / 2
    fig, axes = pair()
    for ax, value, title in zip(axes, [l1, en],
                               ["L1: λ₁ = 0,2", "L1 + L2: λ₂ = 0,8"]):
        contour = ax.contour(U, V, value, levels=[.1, .3, .5],
                             colors=COLORS[:3])
        ax.clabel(contour, fontsize=11)
        ax.axhline(0, color=".6", lw=.7)
        ax.axvline(0, color=".6", lw=.7)
        ax.set_aspect("equal")
        ax.set_title(title)
        mark_axes(ax, "Коэффициент w1", "Коэффициент w2")
    savefig(fig, "f16_penalties")
    X = [[-1., -1.], [1., 1.]]
    y = [-3., 3.]
    strengths = [0., .05, .2, .8, 2.]
    records = []
    for value in strengths:
        alpha = .2 + value
        fit = elasticnet_cd(X, y, alpha=alpha, l1_ratio=.2 / alpha,
                            fit_intercept=False, tol=1e-11)
        records.append({"lambda2": value, "alpha": alpha,
                        "rho": .2 / alpha, "coef": fit["coef"],
                        "objective": fit["objective"],
                        "converged": fit["converged"]})
    fig, axes = pair()
    weights = np.array([row["coef"] for row in records])
    for j in range(2):
        axes[0].plot(strengths[1:], weights[1:, j], "o-", color=COLORS[j],
                     label=f"w{j + 1}")
        axes[0].scatter([0], [weights[0, j]], color=COLORS[j])
    mark_axes(axes[0], "λ₂ при неизменном λ₁ = 0,2", "Коэффициент")
    axes[0].legend()
    idx = np.arange(2)
    axes[1].bar(idx - .2, weights[0], .4, color=COLORS[0],
                label="Lasso: λ₂ = 0")
    axes[1].bar(idx + .2, weights[3], .4, color=COLORS[1],
                label="Elastic Net: λ₂ = 0,8")
    axes[1].set_xticks(idx, ["w1", "w2"])
    mark_axes(axes[1], "Два одинаковых признака", "Коэффициент")
    axes[1].legend(fontsize=11)
    savefig(fig, "f16_grouping")
    assert np.allclose(weights[3], [1, 1], atol=1e-8)
    assert abs(records[3]["objective"] - 1.7) < 1e-9
    assert all(row["converged"] for row in records)
    return finish(16, {"lambda1": .2, "records": records})


RESULT = ch16()
FIGURES = ['f16_penalties', 'f16_grouping']
