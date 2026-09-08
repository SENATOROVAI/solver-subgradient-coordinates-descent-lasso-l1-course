"""Глава 17: воспроизводимый эксперимент."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from plot_utils import savefig
from pure_core import elasticnet_cd
from pure_core import predict
from pure_core import soft_threshold
from experiments_b import (
    finish, pair, mark_axes, standard_data, report_fit, COLORS)


def ch17():
    X, y, truth, Z, stats = standard_data()
    alpha, rho = .2, .4
    fit = elasticnet_cd(Z, y, alpha=alpha, l1_ratio=rho,
                        tol=1e-10, max_iter=30000)
    model = ElasticNet(alpha=alpha, l1_ratio=rho, tol=1e-12,
                       max_iter=30000).fit(Z, y)
    ours = np.array(predict(Z, fit["coef"], fit["intercept"]))
    error = float(np.max(np.abs(ours - model.predict(Z))))
    coef_error = float(np.max(np.abs(fit["coef"] - model.coef_)))
    fig, axes = pair()
    idx = np.arange(8)
    axes[0].bar(idx - .18, fit["coef"], .36, color=COLORS[0],
                label="Наш Python")
    axes[0].bar(idx + .18, model.coef_, .36, color=COLORS[1],
                label="sklearn ElasticNet")
    axes[0].set_xticks(idx, [f"x{i + 1}" for i in idx])
    mark_axes(axes[0], "Стандартизованный признак", "Коэффициент")
    axes[0].legend(fontsize=11)
    axes[1].scatter(model.predict(Z), ours - model.predict(Z),
                    color=COLORS[0], s=15)
    axes[1].axhline(0, color=".4", lw=.7)
    mark_axes(axes[1], "Прогноз sklearn", "Наш прогноз − sklearn")
    savefig(fig, "f17_agreement")
    fig, axes = pair()
    denominators = [1., 1.3, 1.8]
    updates = [soft_threshold(1.5, .3) / value
               for value in denominators]
    axes[0].bar(["λ₂ = 0", "λ₂ = 0,3", "λ₂ = 0,8"], updates,
                color=COLORS[:3])
    mark_axes(axes[0], "z = 1,5; q = 1; λ₁ = 0,3",
              "Новое w = 1,2 / (1 + λ₂)")
    axes[1].plot(range(len(fit["history"])), fit["history"],
                 color=COLORS[0])
    mark_axes(axes[1], "Полные проходы", "Полный критерий Elastic Net")
    savefig(fig, "f17_updates")
    references = [
        ("rho1", .2, 1., Lasso(alpha=.2, tol=1e-12, max_iter=30000)),
        ("rho0", .2, 0., Ridge(alpha=len(y) * .2)),
        ("alpha0", 0., .5, LinearRegression()),
    ]
    endpoints = []
    for name, strength, ratio, reference in references:
        run = elasticnet_cd(Z, y, alpha=strength, l1_ratio=ratio,
                            tol=1e-9, max_iter=30000)
        reference.fit(Z, y)
        prediction = predict(Z, run["coef"], run["intercept"])
        delta = float(np.max(np.abs(prediction - reference.predict(Z))))
        assert delta < 1e-6 and run["converged"]
        endpoints.append(dict(name=name, prediction_error=delta,
                              **report_fit(run)))
    assert error < 1e-6 and coef_error < 1e-6
    return finish(17, {"alpha": alpha, "rho": rho,
                       "lambda1": alpha * rho,
                       "lambda2": alpha * (1 - rho),
                       "prediction_error": error, "coef_error": coef_error,
                       "diagnostic": report_fit(fit),
                       "scalar_updates": updates, "endpoints": endpoints})


RESULT = ch17()
FIGURES = ['f17_agreement', 'f17_updates']
