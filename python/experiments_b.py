"""Experiments 11--20: pure algorithms and labelled library bridge."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from plot_utils import savefig
from pure_core import marketing_data, scale_fit, scale_transform




COLORS = ["#167d9a", "#d16b35", "#7352a3", "#518b44", "#bd4656"]


def finish(number, result):
    """Write exact numerical observations for authors and reruns."""
    path = Path(__file__).parent / f"results_{number:02d}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def pair():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    return fig, axes


def mark_axes(axes, xlabel, ylabel):
    axes.set(xlabel=xlabel, ylabel=ylabel)
    axes.grid(alpha=.2)


def standard_data(seed=42, n=160, noise=1.0, correlation=.95):
    X, y, truth = marketing_data(
        n=n, seed=seed, noise=noise, correlation=correlation)
    stats = scale_fit(X)
    Z = scale_transform(X, stats)
    return X, y, truth, Z, stats


def report_fit(fit):
    keys = ["n_iter", "status", "converged", "optimality_error"]
    return {key: fit[key] for key in keys}


def pipeline(model):
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def folds():
    return KFold(n_splits=5, shuffle=True, random_state=42)


def orthogonal_data():
    X = [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]
    y = [2 * row[0] + .5 * row[1] for row in X]
    return X, y


def cv_protocol_figure(splitter, X):
    codes = np.zeros((5, len(X)))
    for row, (train, validation) in enumerate(splitter.split(X)):
        codes[row, train] = 1
        codes[row, validation] = 2
    fig, ax = plt.subplots(figsize=(9., 3.8))
    from matplotlib.colors import ListedColormap
    ax.imshow(codes, cmap=ListedColormap(["#dceef3", "#efb68e"]),
              aspect="auto", vmin=1, vmax=2, interpolation="nearest")
    ax.set_yticks(range(5), [f"Фолд {i + 1}" for i in range(5)])
    ax.set(xlabel="Номер наблюдения только в обучающей части",
           title="Голубой: обучение; оранжевый: проверка")
    savefig(fig, "f19_cv_protocol")


def raw_coefficients(model, n_features):
    fitted = model.named_steps["model"]
    scaler = model.named_steps["scale"]
    if isinstance(fitted, DummyRegressor):
        return np.zeros(n_features), float(fitted.constant_[0, 0])
    raw = fitted.coef_ / scaler.scale_
    intercept = fitted.intercept_ - np.dot(raw, scaler.mean_)
    return np.asarray(raw), float(intercept)


def family_searches():
    alphas = [.003, .01, .03, .1, .3, 1.]
    return [
        ("Среднее", DummyRegressor(strategy="mean"), {}),
        ("OLS", LinearRegression(), {}),
        ("Lasso", Lasso(max_iter=30000, tol=1e-10),
         {"model__alpha": alphas}),
        ("Ridge", Ridge(),
         {"model__alpha": [.01, .1, 1., 3., 10., 30., 100.]}),
        ("Elastic Net", ElasticNet(max_iter=30000, tol=1e-10),
         {"model__alpha": alphas,
          "model__l1_ratio": [.1, .3, .5, .8, .95]}),
    ]
