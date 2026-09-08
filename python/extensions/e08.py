"""Д08. Потеря одного объекта, среднее и Cython-соглашения."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn._loss.loss import AbsoluteError
from sklearn.metrics import mean_absolute_error
from extra_utils import finish_extra
from plot_utils import savefig

# Данные заданы явно: случайных вычислений здесь нет.
y = np.array([1., 2., 3.])
prediction = np.array([0., 2., 4.])
s = np.array([1., 2., 4.])
loss = AbsoluteError()
gradient, hessian = loss.gradient_hessian(y, prediction)
pointwise = loss.loss(y, prediction)
weighted = loss.loss(y, prediction, sample_weight=s)
average = loss(y, prediction, sample_weight=s)
assert np.allclose(gradient, [-1, -1, 1])
assert np.allclose(hessian, 1)
assert np.isclose(average, 5 / 7)
assert np.isclose(average, mean_absolute_error(
    y, prediction, sample_weight=s))
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
index = np.arange(3)
axes[0].bar(index - .18, pointwise, .36, label="|ошибка|")
axes[0].bar(index + .18, weighted, .36, label="s × |ошибка|")
axes[0].set(xlabel="Номер наблюдения", ylabel="Вклад в сумму",
            xticks=index, xticklabels=["1", "2", "3"])
axes[0].legend()
axes[1].scatter(index, gradient, s=80, label="CyAbsoluteError")
axes[1].scatter(index, np.sign(prediction - y), s=65,
                marker="x", label="np.sign")
axes[1].set(xlabel="Номер наблюдения", ylabel="Выбранный субградиент",
            xticks=index, xticklabels=["1", "2", "3"])
axes[1].legend()
savefig(fig, "e08_loss_gradient")
targets = np.array([1., 2., 9.])
heavy = np.array([1., 1., 5.])
grid = np.linspace(0, 10, 201)
plain_curve = [loss(targets, np.full(3, b)) for b in grid]
heavy_curve = [loss(targets, np.full(3, b), sample_weight=heavy)
               for b in grid]
fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.plot(grid, plain_curve, label="Веса 1, 1, 1")
ax.plot(grid, heavy_curve, label="Веса 1, 1, 5")
ax.scatter([2, 9], [8 / 3, 15 / 7], s=70, zorder=3)
ax.set(xlabel="Постоянный прогноз b", ylabel="Средняя абсолютная ошибка")
ax.legend()
savefig(fig, "e08_weighted_median")
RESULT = finish_extra("e08", {
    "version": sklearn.__version__, "closs": type(loss.closs).__name__,
    "loss": pointwise.tolist(), "weighted_loss": weighted.tolist(),
    "gradient": gradient.tolist(), "hessian": hessian.tolist(),
    "weighted_average": float(average),
    "median": float(loss.fit_intercept_only(targets)),
    "weighted_median": float(loss.fit_intercept_only(targets, heavy)),
})
FIGURES = ["e08_loss_gradient", "e08_weighted_median"]
