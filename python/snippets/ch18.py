"""Глава 18: воспроизводимый эксперимент."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score
from plot_utils import savefig
from pure_core import elasticnet_cd
from experiments_b import (
    finish, pair, mark_axes, standard_data, pipeline, folds, COLORS)


def ch18():
    X, y, truth, Z, stats = standard_data(seed=53, n=120)
    alphas = np.logspace(-2, 0, 7)
    ratios = [.1, .3, .5, .8, 1.]
    fig, axes = pair()
    paths = {}
    for index, ratio in enumerate([.1, .5, .8, 1.]):
        weights = []
        for strength in alphas:
            fit = elasticnet_cd(Z, y, alpha=float(strength),
                                l1_ratio=ratio, tol=1e-8,
                                max_iter=30000)
            assert fit["converged"]
            weights.append(fit["coef"])
        weights = np.array(weights)
        paths[str(ratio)] = weights.tolist()
        axes[0].plot(alphas, weights[:, 0], "o-", color=COLORS[index],
                     label=f"ρ = {ratio:g}")
        active = (np.abs(weights) > 1e-8).sum(axis=1)
        axes[1].plot(alphas, active, "o-", color=COLORS[index],
                     label=f"ρ = {ratio:g}")
    for ax in axes:
        ax.set_xscale("log")
        ax.legend(fontsize=11)
    mark_axes(axes[0], "Общая сила штрафа α", "Коэффициент рекламы w1")
    mark_axes(axes[1], "Общая сила штрафа α", "Число ненулевых весов")
    axes[1].set_yticks(range(4, 9))
    savefig(fig, "f18_paths")
    search = GridSearchCV(
        pipeline(ElasticNet(max_iter=30000, tol=1e-10)),
        {"model__alpha": alphas, "model__l1_ratio": ratios},
        scoring="neg_mean_squared_error", cv=folds(), n_jobs=1,
        error_score="raise")
    search.fit(X, y)
    cv = np.empty((len(ratios), len(alphas)))
    for params, value in zip(search.cv_results_["params"],
                             search.cv_results_["mean_test_score"]):
        row = ratios.index(params["model__l1_ratio"])
        col = int(np.flatnonzero(alphas == params["model__alpha"])[0])
        cv[row, col] = -value
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    heat = ax.imshow(cv, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(alphas)), [f"{v:.3g}" for v in alphas])
    ax.set_yticks(range(len(ratios)), [str(v) for v in ratios])
    for row in range(len(ratios)):
        for col in range(len(alphas)):
            ax.text(col, row, f"{cv[row, col]:.2f}", ha="center",
                    va="center", fontsize=11,
                    color="white" if cv[row, col] > 4.5 else ".15")
    ax.set(xlabel="α", ylabel="ρ",
           title="Средняя MSE: масштабирование внутри каждого из 5 фолдов")
    fig.colorbar(heat, ax=ax, label="Проверочная MSE")
    savefig(fig, "f18_cv_map")
    lambda1 = .15
    strengths = [0., .02, .08, .2, .6, 1.5]
    records = []
    for value in strengths:
        alpha = lambda1 + value
        ratio = lambda1 / alpha
        fit = elasticnet_cd(Z, y, alpha=alpha, l1_ratio=ratio,
                            max_iter=30000, tol=1e-9)
        assert fit["converged"]
        model = pipeline(ElasticNet(alpha=alpha, l1_ratio=ratio,
                                    max_iter=30000, tol=1e-10))
        scores = -cross_val_score(model, X, y, cv=folds(),
                                  scoring="neg_mean_squared_error",
                                  error_score="raise")
        records.append({"lambda2": value, "alpha": alpha, "rho": ratio,
                        "coef": fit["coef"],
                        "cv_mse": float(scores.mean())})
    fig, axes = pair()
    weights = np.array([row["coef"] for row in records])
    for j, label, color in [(0, "Реклама x1", COLORS[0]),
                             (4, "Коррелирующий x5", COLORS[1])]:
        axes[0].plot(strengths, weights[:, j], "o-", color=color,
                     label=label)
    mark_axes(axes[0], "λ₂ при неизменном λ₁ = 0,15", "Вес после масштаба")
    axes[0].legend(fontsize=11)
    axes[1].plot(strengths, [row["cv_mse"] for row in records],
                 "o-", color=COLORS[0])
    mark_axes(axes[1], "λ₂ при неизменном λ₁ = 0,15",
              "Средняя MSE на 5 фолдах")
    savefig(fig, "f18_fixed_l1")
    return finish(18, {"seed": 53, "n": 120, "alphas": alphas.tolist(),
                       "ratios": ratios, "cv_mse": cv.tolist(),
                       "best_params": search.best_params_,
                       "best_cv_mse": float(-search.best_score_),
                       "fixed_lambda1": lambda1, "fixed_l1": records,
                       "paths": paths})


RESULT = ch18()
FIGURES = ['f18_paths', 'f18_cv_map', 'f18_fixed_l1']
