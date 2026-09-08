"""Д25. Проследить нормировку и проверить вычислительные ветви."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from source_extra import source_identity, branch_comparison, dual_pair
from pure_core import objective
from extra_utils import finish_extra
from plot_utils import savefig

rng = np.random.default_rng(621)
X = rng.normal(size=(90, 5))
X[:, 4] = .9 * X[:, 0] + .15 * rng.normal(size=90)
X -= X.mean(axis=0)
y = X @ np.array([2., -1.5, .4, 0., 0.])
y += rng.normal(scale=.6, size=90)
y -= y.mean()
identity = source_identity()
comparison = branch_comparison(X, y)
assert max(comparison["branch_errors"].values()) < 1e-6
assert abs(comparison["gap"] - comparison["private_gap_scaled"]) < 1e-9
zero = ElasticNet(alpha=100, fit_intercept=False).fit(
    np.eye(3), np.ones(3))
# Отдельный исторический пример: квадрат нормы столбца равен 1.
XP = np.array([[-1.], [1.]]) / np.sqrt(2)
yp = 3 * XP[:, 0]
naive = ElasticNet(alpha=.5, l1_ratio=.2, fit_intercept=False,
                   tol=1e-12).fit(XP, yp).coef_
corrected = (1 + 2 * .4) * naive
historical = {"naive": float(naive[0]),
              "corrected": float(corrected[0]),
              "J_naive": objective(XP, yp, naive, 0., .5, .2),
              "J_corrected": objective(XP, yp, corrected, 0., .5, .2)}
assert np.isclose(naive[0], 14 / 9)
assert np.isclose(corrected[0], 2.8)
assert historical["J_naive"] < historical["J_corrected"]
certificates = [dual_pair([[-1., -1.], [1., 1.]], [-3., 3.], w)
                for w in [[0., 0.], [1., 1.]]]
assert np.isclose(certificates[0]["gap_A"], 3.92)
assert np.isclose(certificates[0]["gap_B"], 9.8)
assert abs(certificates[1]["gap_A"]) < 1e-12
assert abs(certificates[1]["gap_B"]) < 1e-12
mutable_X = np.asfortranarray(X + 3.)
before = mutable_X.copy()
ElasticNet(alpha=.2, l1_ratio=.6, copy_X=False).fit(mutable_X, y)
input_change = float(np.max(np.abs(mutable_X - before)))
assert input_change > 1.
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
names = list(comparison["branch_errors"])
values = list(comparison["branch_errors"].values())
axes[0].bar(names, values)
axes[0].tick_params(axis="x", rotation=30)
axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axes[0].set(ylabel="Максимальное расхождение прогнозов")
index = np.arange(5)
axes[1].bar(index - .18, comparison["coef"], .36,
             label="Без ограничения")
axes[1].bar(index + .18, comparison["positive_coef"], .36,
             label="w ≥ 0")
axes[1].set(xlabel="Признак", ylabel="Коэффициент",
            xticks=index, xticklabels=[f"x{i+1}" for i in index])
axes[1].legend(fontsize=11)
savefig(fig, "e25_branches")
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
for ax, name, title in zip(axes, ["independent", "joint"],
                          ["Независимые L1", "Совместный L21"]):
    W = np.array(comparison[name])
    for task in range(2):
        ax.plot(index, W[task], "o-", label=f"Задача {task+1}")
    ax.set(title=title, xlabel="Признак", ylabel="Коэффициент",
           xticks=index, xticklabels=[f"x{i+1}" for i in index])
    ax.legend()
savefig(fig, "e25_multioutput")
RESULT = finish_extra("e25", {
    "source": identity, **comparison,
    "initial_stop_iterations": int(zero.n_iter_),
    "initial_stop_gap": float(zero.dual_gap_),
    "penalties_internal": [90 * .2 * .6, 90 * .2 * .4],
    "historical": historical,
    "dual_pairs_zero_optimum": certificates,
    "copy_X_false_input_change": input_change,
})
FIGURES = ["e25_branches", "e25_multioutput"]
