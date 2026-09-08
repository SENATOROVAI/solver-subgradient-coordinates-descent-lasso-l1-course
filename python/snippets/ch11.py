"""Глава 11: воспроизводимый эксперимент."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from plot_utils import savefig
from pure_core import soft_threshold
from experiments_b import (
    finish, mark_axes, COLORS)


def ch11():
    threshold = .8
    z = np.linspace(-2.5, 2.5, 501)
    values = [soft_threshold(float(v), threshold) for v in z]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.axvspan(-threshold, threshold, color=COLORS[0], alpha=.12)
    ax.plot(z, z, "--", color=".6", label="Без штрафа: w = z")
    ax.plot(z, values, color=COLORS[0], lw=2.5,
            label="Мягкий порог: λ₁ = 0,8")
    ax.axhline(0, color=".4", lw=.8)
    ax.axvline(0, color=".4", lw=.8)
    mark_axes(ax, "Значение z до порога", "Коэффициент после порога")
    ax.legend()
    savefig(fig, "f11_soft")
    cases = [-1.8, .4, 1.8]
    minima = [soft_threshold(v, threshold) for v in cases]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    w = np.linspace(-2.5, 2.5, 801)
    for k, (value, minimum) in enumerate(zip(cases, minima)):
        curve = .5 * (w - value) ** 2 + threshold * np.abs(w)
        score = .5 * (minimum - value) ** 2 + threshold * abs(minimum)
        ax.plot(w, curve, color=COLORS[k], label=f"z = {value:g}")
        ax.scatter([minimum], [score], color=COLORS[k], s=60, zorder=5)
    mark_axes(ax, "Пробный коэффициент w", "(w − z)² / 2 + 0,8 |w|")
    ax.set_ylim(0, 5)
    ax.legend()
    savefig(fig, "f11_scalar")
    assert np.allclose(minima, [-1, 0, 1], atol=1e-12)
    return finish(11, {"lambda1": threshold, "q": 1,
                       "z": cases, "minima": minima})


RESULT = ch11()
FIGURES = ['f11_soft', 'f11_scalar']
