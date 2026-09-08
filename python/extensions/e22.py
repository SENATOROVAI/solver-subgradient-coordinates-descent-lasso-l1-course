"""Д22. Веса, центрирование и повторение наблюдений."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from pure_core import marketing_data, coordinate_descent, predict
from weighted_extra import weighted_cd, weighted_objective
from extra_utils import finish_extra
from plot_utils import savefig

X, y, _ = marketing_data(n=60, p=5, seed=619, noise=2.)
y[0] += 15
s = [9.] + [1.] * 59
settings = dict(alpha=.2, l1_ratio=.6, tol=1e-10)
fit = weighted_cd(X, y, s, **settings)
scaled = weighted_cd(X, y, [17 * v for v in s], **settings)
plain = coordinate_descent(X, y, **settings)
reference = ElasticNet(**dict(settings, tol=1e-12),
                       max_iter=30000).fit(X, y, sample_weight=s)
XR, yr = [], []
for row, target, copies in zip(X, y, s):
    XR.extend([row] * int(copies))
    yr.extend([target] * int(copies))
repeated = coordinate_descent(XR, yr, **settings)
ours = np.array(predict(X, fit["coef"], fit["intercept"]))
errors = {}
for name, other in [("scale17", scaled), ("replication", repeated)]:
    errors[name] = float(np.max(np.abs(ours - predict(
        X, other["coef"], other["intercept"]))))
errors["sklearn"] = float(np.max(np.abs(ours - reference.predict(X))))
assert max(errors.values()) < 1e-6 and fit["converged"]
value = weighted_objective(
    X, y, s, fit["coef"], fit["intercept"], .2, .6)
assert abs(value - fit["objective"]) < 1e-10
fig, ax = plt.subplots(figsize=(7.4, 4.0))
index = np.arange(5)
ax.bar(index - .18, plain["coef"], .36, label="Все веса равны 1")
ax.bar(index + .18, fit["coef"], .36, label="Первый вес равен 9")
ax.set(xlabel="Признак", ylabel="Коэффициент в исходном масштабе",
       xticks=index, xticklabels=[f"x{i+1}" for i in index])
ax.legend()
savefig(fig, "e22_weighted_coefficients")
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
axes[0].plot(fit["history"])
axes[0].set(xlabel="Полный проход", ylabel="Взвешенная цель J")
axes[1].scatter(range(60), y - ours, s=np.array(s) * 13,
                alpha=.75)
axes[1].axhline(0, color=".4", lw=.8)
axes[1].set(xlabel="Номер наблюдения", ylabel="Остаток y − прогноз")
savefig(fig, "e22_weighted_history")
RESULT = finish_extra("e22", {
    "errors": errors, "objective": value,
    "coef": fit["coef"], "intercept": fit["intercept"],
    "n_iter": fit["n_iter"], "kkt": fit["optimality_error"],
    "plain_first_residual": y[0] - predict(
        X, plain["coef"], plain["intercept"])[0],
    "weighted_first_residual": float(y[0] - ours[0]),
    "normalized_weight_sum": fit["weight_sum_normalized"],
    "replicated_rows": len(yr),
})
FIGURES = ["e22_weighted_coefficients", "e22_weighted_history"]
