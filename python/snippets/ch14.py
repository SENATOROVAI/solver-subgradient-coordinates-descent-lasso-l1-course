"""Глава 14: воспроизводимый эксперимент."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import Lasso
from plot_utils import savefig
from pure_core import lasso_cd
from experiments_b import (
    finish, pair, mark_axes, COLORS)


def ch14():
    X = [[-1., -1.], [1., 1.]]
    y = [-3., 3.]
    first = lasso_cd(X, y, alpha=.2, fit_intercept=False, w0=[0., 0.])
    second = lasso_cd(X, y, alpha=.2, fit_intercept=False, w0=[0., 3.])
    fig, axes = pair()
    idx = np.arange(2)
    axes[0].bar(idx - .2, first["coef"], .4, color=COLORS[0],
                label="Старт (0; 0)")
    axes[0].bar(idx + .2, second["coef"], .4, color=COLORS[1],
                label="Старт (0; 3)")
    axes[0].set_xticks(idx, ["w1", "w2"])
    axes[0].set_ylabel("Коэффициент")
    axes[0].legend(fontsize=11)
    grid = np.linspace(-1, 1, 100)
    for fit, color, style in [(first, COLORS[0], "-"),
                              (second, COLORS[1], "--")]:
        axes[1].plot(grid, sum(fit["coef"]) * grid, style,
                     color=color, lw=2.5)
    axes[1].scatter([-1, 1], y, color=".2", label="Наблюдения")
    mark_axes(axes[1], "Одинаковые x1 = x2", "Прогноз: обе линии совпали")
    axes[1].legend(fontsize=11)
    savefig(fig, "f14_duplicates")
    rng = np.random.default_rng(140)
    common = rng.normal(size=80)
    target = 3 * common + rng.normal(scale=.2, size=80)
    coefficients = []
    predictions = []
    diagnostics = []
    probe = np.linspace(-2, 2, 81)
    probe_X = np.column_stack([probe, probe])
    for _ in range(100):
        matrix = np.column_stack([
            common + rng.normal(scale=.03, size=80),
            common + rng.normal(scale=.03, size=80)])
        fitted = Lasso(alpha=.2, tol=1e-10, max_iter=30000)
        fitted.fit(matrix, target)
        coefficients.append(fitted.coef_)
        predictions.append(fitted.predict(probe_X))
        diagnostics.append(fitted.n_iter_ < 30000)
    coefficients = np.array(coefficients)
    predictions = np.array(predictions)
    sums = coefficients.sum(axis=1)
    fig, axes = pair()
    axes[0].boxplot([coefficients[:, 0], coefficients[:, 1], sums],
                    tick_labels=["w1", "w2", "w1 + w2"],
                    showfliers=False)
    mark_axes(axes[0], "100 возмущений признаков", "Коэффициент")
    for prediction in predictions:
        axes[1].plot(probe, prediction, color=COLORS[0], alpha=.08)
    axes[1].plot(probe, 3 * probe, "--", color=COLORS[1],
                 label="Среднее y в генераторе")
    mark_axes(axes[1], "Общие точки x1 = x2", "100 прогнозов с перехватом")
    axes[1].legend(fontsize=11)
    savefig(fig, "f14_stability")
    assert abs(first["objective"] - .58) < 1e-10
    assert abs(second["objective"] - .58) < 1e-10
    assert all(diagnostics)
    return finish(14, {"first": first["coef"], "second": second["coef"],
                       "objective_both": first["objective"],
                       "perturbation_seed": 140, "repeats": 100,
                       "feature_noise_sd": .03, "target_noise_sd": .2,
                       "coef_mean": coefficients.mean(axis=0).tolist(),
                       "coef_sd": coefficients.std(axis=0).tolist(),
                       "sum_mean": float(sums.mean()),
                       "sum_sd": float(sums.std()),
                       "prediction_sd_at_x1": float(
                           predictions[:, 60].std()),
                       "max_prediction_sd": float(predictions.std(0).max()),
                       "probe": probe.tolist(),
                       "prediction_q10": np.quantile(
                           predictions, .1, axis=0).tolist(),
                       "prediction_q90": np.quantile(
                           predictions, .9, axis=0).tolist(),
                       "all_converged": all(diagnostics)})


RESULT = ch14()
FIGURES = ['f14_duplicates', 'f14_stability']
