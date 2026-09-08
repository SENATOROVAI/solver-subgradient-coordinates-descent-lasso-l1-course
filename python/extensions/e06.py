"""Д06. LP-постановка LAD и условные квантили доставки."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from sklearn.linear_model import QuantileRegressor
from pure_core import mae
from mae_extra import pinball, quantile_interval
from plot_utils import savefig
from extra_utils import finish_extra

FIGURES = ["e06_pinball_constant", "e06_quantile_predictions"]
# Независимая линейная программа: переменные w, b, u_1,...,u_n.
x, target = np.array([0., 1., 2., 3.]), np.array([1., 2., 2., 5.])
n = len(x)
left = np.column_stack([-x, -np.ones(n), -np.eye(n)])
right = np.column_stack([x, np.ones(n), -np.eye(n)])
lp = linprog(np.r_[0., 0., np.ones(n) / n],
             A_ub=np.vstack([left, right]), b_ub=np.r_[-target, target],
             bounds=[(None, None)] * 2 + [(0, None)] * n,
             method="highs")
assert lp.success
reference = QuantileRegressor(quantile=.5, alpha=0,
                              solver="highs").fit(x[:, None], target)
assert abs(lp.fun - mae(target, reference.predict(x[:, None]))) < 1e-9
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.3))
r = np.linspace(-4, 4, 201)
for q in [.25, .5, .9]:
    axes[0].plot(r, np.maximum(q*r, (q-1)*r), label=f"q={q}")
axes[0].set(xlabel="Остаток y − прогноз", ylabel="Pinball loss")
axes[0].legend()
grid = np.linspace(0, 30, 301)
ys = [0, 10, 20, 30]
axes[1].plot(grid, [pinball(ys, [a]*4, .6) for a in grid])
axes[1].scatter([18, 20], [4.7, 4.5], s=65)
axes[1].annotate("18: интерполяция", (18, 4.7), (1, 7.8),
                 arrowprops={"arrowstyle": "->"})
axes[1].annotate("20: минимум", (20, 4.5), (18, 6.2),
                 arrowprops={"arrowstyle": "->"})
axes[1].set(xlabel="Постоянный прогноз", ylabel="Средняя потеря, q=0.6")
savefig(fig, FIGURES[0])
rng = np.random.default_rng(104)
train_x = rng.uniform(0, 10, size=180)
test_x = rng.uniform(0, 10, size=500)
def responses(values):
    return 2 + 1.5*values + (.25 + .2*values)*rng.normal(size=len(values))
train_y, test_y = responses(train_x), responses(test_x)
line_x = np.linspace(0, 10, 201)
fig, ax = plt.subplots(figsize=(7.8, 4.7))
ax.scatter(train_x, train_y, s=13, color="#A2A8AE", alpha=.55,
           label="Обучающие доставки")
models, predictions, metrics = [], [], {}
for q in [.1, .5, .9]:
    model = QuantileRegressor(quantile=q, alpha=0,
                              solver="highs").fit(train_x[:, None], train_y)
    models.append(model)
    pred = model.predict(test_x[:, None])
    predictions.append(pred)
    ax.plot(line_x, model.predict(line_x[:, None]), lw=2, label=f"q={q}")
    metrics[str(q)] = {"pinball_test": pinball(test_y, pred, q),
                       "fraction_below": float(
                           np.mean(test_y <= pred))}
ax.set(xlabel="Расстояние, условные единицы", ylabel="Время доставки")
ax.legend()
savefig(fig, FIGURES[1])
inside = (test_y >= predictions[0]) & (test_y <= predictions[2])
coverage = {"all": float(inside.mean()),
            "short": float(inside[test_x < 5].mean()),
            "long": float(inside[test_x >= 5].mean())}
lines = np.array([model.predict(line_x[:, None]) for model in models])
crossings = int(np.sum(
    np.any(np.diff(lines, axis=0) < 0, axis=0)))
# Экстраполяционная проверка: порядок не обеспечен общим обучением.
wide_x = np.linspace(-10, 20, 601)
wide = np.array([model.predict(wide_x[:, None]) for model in models])
wide_crossings = int(np.sum(
    np.any(np.diff(wide, axis=0) < 0, axis=0)))
assert quantile_interval(ys, .6) == (20, 20)
assert abs(pinball(ys, [18]*4, .6) - 4.7) < 1e-12
assert abs(2 * pinball(target, [0]*n, .5) - mae(target, [0]*n)) < 1e-12
RESULT = {"lp_lad_mae": float(lp.fun), "lp_coef": lp.x[:2].tolist(),
          "median_reference_mae": mae(
              target, reference.predict(x[:, None])),
          "empirical_quantile": 20, "linear_interpolation": 18,
          "pinball_at_20": 4.5, "pinball_at_18": 4.7,
          "coverage": coverage, "quantile_test_metrics": metrics,
          "crossing_grid_points": crossings,
          "extrapolation_crossing_grid_points": wide_crossings,
          "median_baseline_test_mae": mae(
              test_y, [float(np.median(train_y))]*len(test_y))}
finish_extra("e06", RESULT)
