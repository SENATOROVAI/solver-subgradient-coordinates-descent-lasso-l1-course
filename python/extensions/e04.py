"""Д04. Число обработанных строк и случайный оцениватель."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import random
import statistics
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor
from pure_core import mae
from mae_extra import stochastic_mae
from plot_utils import savefig
from extra_utils import finish_extra

FIGURES = ["e04_equal_budget", "e04_weighted_estimators"]
rng = random.Random(103)
X = [[-2 + 4 * i / 59] for i in range(60)]
y = [1 + 2 * row[0] + rng.gauss(0, .35) for row in X]
reference = QuantileRegressor(quantile=.5, alpha=0,
                              solver="highs").fit(X, y)
reference_loss = mae(y, reference.predict(X).tolist())
budget = 24000
runs = {}
fig, ax = plt.subplots(figsize=(7.8, 4.5))
for name, size, sampling in [("Полный", 60, "full"),
                              ("Mini-batch 10", 10, "uniform"),
                              ("SGD", 1, "uniform")]:
    updates = budget // size
    fit = stochastic_mae(X, y, batch_size=size, sampling=sampling,
                         updates=updates,
                         evaluate_every=max(1, updates//80),
                         step=.3, power=.6, seed=103)
    runs[name] = fit
    ax.plot([r["rows"] for r in fit["history"]],
            [r["loss"] for r in fit["history"]], label=name)
ax.axhline(reference_loss, color="black", ls="--", label="LAD-ориентир")
ax.set(xlabel="Строки, использованные для обновлений",
       ylabel="Фактическая MAE в моменты проверки", yscale="log")
ax.legend()
savefig(fig, FIGURES[0])
# Два сырых вклада: -1 и 2, веса объектов: 1 и 3.
raw, weights = [-1.0, 2.0], [1, 3]
means, deviations = [], []
for size in [1, 4, 16]:
    estimates = []
    for _ in range(3000):
        ids = [rng.randrange(2) for _ in range(size)]
        estimates.append(sum(2 * weights[i] / 4 * raw[i]
                             for i in ids) / size)
    means.append(statistics.mean(estimates))
    deviations.append(statistics.pstdev(estimates))
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.errorbar([1, 4, 16], means, yerr=deviations, marker="o",
            capsize=6, label="Среднее ± SD случайной оценки")
ax.axhline(1.25, color="black", ls="--", label="Полный взвешенный вклад")
ax.set(xlabel="Размер случайного пакета", ylabel="Оценка субградиента по w",
       xticks=[1, 4, 16])
ax.legend()
savefig(fig, FIGURES[1])
assert sum([-.5, 3]) / 2 == 1.25
assert .25 * (-1) + .75 * 2 == 1.25
assert all(f["update_rows"] == budget for f in runs.values())
RESULT = {"reference_mae": reference_loss, "update_budget": budget,
          "runs": {name: {key: fit[key] for key in
                    ["best_loss", "last_loss", "average_loss", "n_iter",
                     "update_rows", "diagnostic_rows", "status"]}
                   for name, fit in runs.items()},
          "weighted_full_gradient": 1.25,
          "uniform_corrected_values": [-.5, 3],
          "probability_sampling_values": [-1, 2],
          "wrong_double_weight_expectation": 4.25,
          "batch_means": means, "batch_standard_deviations": deviations}
finish_extra("e04", RESULT)
