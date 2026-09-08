"""Глава 06. Выброс y и точка с экстремальным x."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import matplotlib.pyplot as plt
from pure_core import predict, mae, mse, fit_mae
from plot_utils import savefig
from experiments_a import finish
# Библиотечный эталон MAE: медианная квантильная регрессия.
from sklearn.linear_model import LinearRegression, QuantileRegressor

X = [[-5 + i * .5] for i in range(21)]
y = [2 * row[0] + 1 + .35 * math.sin(i * 1.7)
     for i, row in enumerate(X)]
y_out = y.copy()
y_out[10] += 25
high_X, high_y = X + [[60]], y + [-20]
cases = [("Исходные данные", X, y), ("Выброс по y", X, y_out),
         ("Большое x", high_X, high_y)]
RESULT = {"cases": [], "base_n": 21, "y_outlier_increment": 25,
          "leverage_point": [60, -20]}
fits = []
for label, rows, target in cases:
    models = [QuantileRegressor(quantile=.5, alpha=0, solver="highs"),
              LinearRegression()]
    row = {"name": label, "models": {}}
    for name, model in zip(["MAE", "OLS"], models):
        model.fit(rows, target)
        coef, intercept = model.coef_.tolist(), float(model.intercept_)
        pred = predict(rows, coef, intercept)
        row["models"][name] = {"coef": coef, "intercept": intercept,
                                "mae": mae(target, pred),
                                "mse": mse(target, pred),
                                "base_mae": mae(y, predict(X, coef,
                                                           intercept))}
    fits.append(models)
    RESULT["cases"].append(row)
own = fit_mae(X, y_out, steps=12000, step=.12, decay=.6,
              w0=[0], b0=0)
RESULT["own_y_outlier"] = {k: own[k] for k in
                           ["coef", "intercept", "best_loss", "status"]}
FIGURES = ["f06_outlier_y", "f06_leverage"]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
for idx, ax in enumerate(axes):
    label, rows, target = cases[idx]
    ax.scatter([r[0] for r in rows], target, s=24, color="black")
    for name, model in zip(["MAE", "OLS"], fits[idx]):
        xx = [[-5.5], [5.5]]
        ax.plot([r[0] for r in xx], model.predict(xx), label=name)
    if idx == 1:
        ax.plot([-5.5, 5.5], predict([[-5.5], [5.5]], own["coef"],
                                   own["intercept"]), ls=":", lw=2,
                label="Наш MAE:\nлучший шаг")
    ax.set(xlabel="Признак x", ylabel="Отклик y", title=label,
           xlim=(-5.7, 5.7), ylim=(-11, 29))
    ax.legend(fontsize=11, loc="upper left")
savefig(fig, FIGURES[0])

fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
for ax, limits, title in [(axes[0], (-6, 62), "Все наблюдения"),
                          (axes[1], (-5.7, 5.7), "Ближе: обычные x")]:
    ax.scatter([r[0] for r in high_X], high_y, s=26, c="black")
    xx = [[limits[0]], [limits[1]]]
    ax.plot([r[0] for r in xx], [2 * r[0] + 1 for r in xx],
            ls="--", color="#999999", label="Исходный закон")
    for name, model in zip(["MAE", "OLS"], fits[2]):
        ax.plot([r[0] for r in xx], model.predict(xx), label=name)
    ax.set(xlabel="Признак x", ylabel="Отклик y", title=title,
           xlim=limits, ylim=(-24, 18) if ax is axes[0] else (-12, 14))
    ax.legend(fontsize=11)
savefig(fig, FIGURES[1])
finish(6, RESULT)
