"""Глава 13: воспроизводимый эксперимент."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from plot_utils import savefig
from pure_core import coordinate_descent, lasso_cd
from pure_core import predict
from experiments_b import (
    finish, pair, mark_axes, standard_data, report_fit, COLORS)


def ch13():
    X, y, truth, Z, stats = standard_data()
    fit = lasso_cd(Z, y, alpha=.15, tol=1e-9, max_iter=30000)
    library = Lasso(alpha=.15, tol=1e-12, max_iter=30000).fit(Z, y)
    ours = np.array(predict(Z, fit["coef"], fit["intercept"]))
    theirs = library.predict(Z)
    weight_error = float(np.max(np.abs(fit["coef"] - library.coef_)))
    prediction_error = float(np.max(np.abs(ours - theirs)))
    fig, axes = pair()
    idx = np.arange(8)
    axes[0].bar(idx - .18, fit["coef"], .36, color=COLORS[0],
                label="Наш Python")
    axes[0].bar(idx + .18, library.coef_, .36, color=COLORS[1],
                label="sklearn Lasso")
    axes[0].set_xticks(idx, [f"x{i + 1}" for i in idx])
    mark_axes(axes[0], "Стандартизованный признак", "Коэффициент")
    axes[0].legend(fontsize=11)
    axes[1].scatter(theirs, ours - theirs, s=13, color=COLORS[0])
    axes[1].axhline(0, color=".4", lw=1)
    mark_axes(axes[1], "Прогноз sklearn", "Наш прогноз − sklearn")
    savefig(fig, "f13_agreement")
    budgets = [1, 10, 100, 3000]
    runs = [lasso_cd(Z, y, alpha=.15, tol=1e-9, max_iter=n)
            for n in budgets]
    fig, axes = pair()
    for k, (budget, run) in enumerate(zip(budgets, runs)):
        axes[0].plot(range(len(run["history"])), run["history"],
                     color=COLORS[k], label=f"Лимит {budget}")
    axes[0].set_xscale("symlog", linthresh=1)
    mark_axes(axes[0], "Полные проходы", "Полный критерий J")
    axes[0].legend(fontsize=11)
    errors = [run["optimality_error"] for run in runs]
    axes[1].bar([str(n) for n in budgets], errors, color=COLORS[0])
    axes[1].axhline(1e-9, color=COLORS[1], ls="--", label="Допуск 10⁻⁹")
    axes[1].set_yscale("log")
    mark_axes(axes[1], "Лимит проходов", "Ошибка проверки минимума")
    axes[1].legend(fontsize=11)
    savefig(fig, "f13_budget")
    zero = lasso_cd([[0., 1.], [0., 2.], [0., 3.]], [3., 5., 8.],
                    alpha=.1)
    plain = coordinate_descent(Z, y, alpha=0., tol=1e-9,
                               max_iter=30000)
    ols = LinearRegression().fit(Z, y)
    plain_pred = predict(Z, plain["coef"], plain["intercept"])
    plain_error = float(np.max(np.abs(plain_pred - ols.predict(Z))))
    assert weight_error < 1e-6 and prediction_error < 1e-6
    assert zero["coef"][0] == 0 and plain_error < 1e-6
    return finish(13, {"alpha": .15, "coef_error": weight_error,
                       "prediction_error": prediction_error,
                       "diagnostic": report_fit(fit),
                       "budgets": [dict(budget=n, **report_fit(run))
                                   for n, run in zip(budgets, runs)],
                       "zero_column_coef": zero["coef"],
                       "alpha0_prediction_error": plain_error,
                       "alpha0_diagnostic": report_fit(plain)})


RESULT = ch13()
FIGURES = ['f13_agreement', 'f13_budget']
