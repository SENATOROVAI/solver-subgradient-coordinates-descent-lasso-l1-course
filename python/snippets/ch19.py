"""Глава 19: воспроизводимый эксперимент."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from plot_utils import savefig
from pure_core import marketing_data
from experiments_b import (
    finish, pair, mark_axes, pipeline, folds, cv_protocol_figure, COLORS)


def ch19():
    X, y, truth = marketing_data(n=200, seed=77, noise=2.,
                                  correlation=.9)
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=.25, random_state=42)
    cv_protocol_figure(folds(), train_X)
    search = GridSearchCV(
        pipeline(ElasticNet(max_iter=30000, tol=1e-10)),
        {"model__alpha": [.01, .03, .1, .3, 1.],
         "model__l1_ratio": [.2, .5, .8, 1.]},
        cv=folds(), scoring="neg_mean_squared_error", n_jobs=1,
        error_score="raise")
    search.fit(train_X, train_y)
    chosen_params = dict(search.best_params_)
    chosen_model = search.best_estimator_
    test_prediction = chosen_model.predict(test_X)
    test_mse = mean_squared_error(test_y, test_prediction)
    test_mae = mean_absolute_error(test_y, test_prediction)
    fig, axes = pair()
    axes[0].scatter(test_y, test_prediction, s=23, color=COLORS[0])
    limits = [min(test_y), max(test_y)]
    axes[0].plot(limits, limits, "--", color=".5")
    mark_axes(axes[0], "Фактическое y на финальном тесте", "Прогноз")
    errors = np.array(test_y) - test_prediction
    axes[1].hist(errors, bins=12, color=COLORS[0], edgecolor="white")
    axes[1].axvline(0, color=COLORS[1], lw=2)
    mark_axes(axes[1], "Остаток y − прогноз", "Число наблюдений теста")
    savefig(fig, "f19_test")
    return finish(19, {"seed": 77, "n_train": len(train_y),
                       "n_test": len(test_y), "best_params": chosen_params,
                       "cv_mse": float(-search.best_score_),
                       "test_mse": float(test_mse),
                       "test_mae": float(test_mae),
                       "selection_before_test": True,
                       "fold_train_size": 120, "fold_validation_size": 30})


RESULT = ch19()
FIGURES = ['f19_cv_protocol', 'f19_test']
