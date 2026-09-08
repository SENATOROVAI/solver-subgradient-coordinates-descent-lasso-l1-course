"""Глава 20: воспроизводимый эксперимент."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from plot_utils import savefig
from pure_core import marketing_data
from experiments_b import (
    finish, pair, mark_axes, pipeline, folds, raw_coefficients,
    family_searches, COLORS)


def ch20():
    X, y, truth = marketing_data(n=160, seed=42)
    train_X, test_X, train_y, test_y = train_test_split(
        np.array(X), np.array(y), test_size=.25, random_state=42)
    trained = []
    records = []
    for name, model, parameters in family_searches():
        search = GridSearchCV(
            pipeline(model), parameters, cv=folds(), n_jobs=1,
            scoring={"mse": "neg_mean_squared_error",
                     "mae": "neg_mean_absolute_error"}, refit="mse",
            error_score="raise")
        search.fit(train_X, train_y)
        chosen = search.best_index_
        cv = search.cv_results_
        candidates = []
        for i, params in enumerate(cv["params"]):
            candidates.append({"params": params,
                               "cv_mse": float(-cv["mean_test_mse"][i]),
                               "cv_mae": float(-cv["mean_test_mae"][i])})
        records.append({"name": name, "params": search.best_params_,
                        "cv_mse": float(-cv["mean_test_mse"][chosen]),
                        "cv_mae": float(-cv["mean_test_mae"][chosen]),
                        "cv_mse_sd": float(cv["std_test_mse"][chosen]),
                        "candidates": candidates})
        trained.append(search.best_estimator_)
    winner_index = min(range(len(records)),
                       key=lambda i: records[i]["cv_mse"])
    winner_name = records[winner_index]["name"]
    winner = trained[winner_index]
    rng = np.random.default_rng(42)
    weights = []
    intercepts = []
    for _ in range(100):
        sample = rng.choice(len(train_y), size=96, replace=False)
        fitted = clone(winner).fit(train_X[sample], train_y[sample])
        coef, intercept = raw_coefficients(fitted, train_X.shape[1])
        weights.append(coef)
        intercepts.append(intercept)
    weights = np.array(weights)
    assert np.all(np.isfinite(weights))
    frequency = (np.abs(weights) > 1e-8).mean(axis=0)
    lower, center, upper = np.quantile(weights, [.1, .5, .9], axis=0)
    for row, model in zip(records, trained):
        predictions = model.predict(test_X)
        row["test_mse"] = float(mean_squared_error(test_y, predictions))
        row["test_mae"] = float(mean_absolute_error(test_y, predictions))
        raw, intercept = raw_coefficients(model, train_X.shape[1])
        assert np.allclose(predictions, test_X @ raw + intercept,
                           atol=1e-10, rtol=1e-12)
        row["raw_coef"] = raw.tolist()
        row["raw_intercept"] = intercept
    fig, axes = pair()
    idx = np.arange(5)
    labels = [row["name"] for row in records]
    for ax, metric in zip(axes, ["mse", "mae"]):
        ax.bar(idx - .18, [row[f"cv_{metric}"] for row in records],
               .36, color=COLORS[0], label="CV: 5 фолдов")
        ax.bar(idx + .18, [row[f"test_{metric}"] for row in records],
               .36, color=COLORS[1], label="Финальный тест")
        ax.set_xticks(idx, labels, rotation=20, ha="right")
        ax.set_yscale("log")
        mark_axes(ax, "Семейство модели", metric.upper() + ": лог. шкала")
        ax.legend(fontsize=11)
    fig.suptitle(f"Семейство выбрано по CV до теста: {winner_name}")
    savefig(fig, "f20_project_scores")
    fig, axes = pair()
    idx = np.arange(8)
    axes[0].barh(idx, frequency, color=COLORS[0])
    axes[0].set_yticks(idx, [f"x{i + 1}" for i in idx])
    axes[0].set_xlim(0, 1.05)
    mark_axes(axes[0], "Доля ненулевых весов\n100 обучающих подвыборок",
              "Признак")
    axes[1].errorbar(center, idx, xerr=[center - lower, upper - center],
                     fmt="o", color=COLORS[0], capsize=3,
                     label="Медиана и квантили 10–90%")
    axes[1].scatter(truth, idx, marker="x", color=COLORS[1], s=50,
                    label="Вес в генераторе")
    axes[1].set_yticks(idx, [f"x{i + 1}" for i in idx])
    axes[1].axvline(0, color=".5", lw=.8)
    mark_axes(axes[1], "Коэффициент в исходных единицах", "Признак")
    axes[1].legend(fontsize=11)
    fig.suptitle(f"Устойчивость {winner_name}: обучающие подвыборки")
    savefig(fig, "f20_project_stability")
    return finish(20, {"seed": 42, "n_train": len(train_y),
                       "n_test": len(test_y), "winner": winner_name,
                       "selection_before_test": True, "models": records,
                       "stability_repeats": 100, "subsample_size": 96,
                       "stability_seed": 42,
                       "stability_frequency": frequency.tolist(),
                       "stability_median": center.tolist(),
                       "stability_q10": lower.tolist(),
                       "stability_q90": upper.tolist(),
                       "stability_raw_coefficients": weights.tolist(),
                       "stability_raw_intercepts": intercepts,
                       "truth_raw_coefficients": truth})


RESULT = ch20()
FIGURES = ['f20_project_scores', 'f20_project_stability']
