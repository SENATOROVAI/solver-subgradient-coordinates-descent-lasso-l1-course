"""Д23. Путь, тёплый старт и правило одной стандартной ошибки."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from pure_core import marketing_data, coordinate_descent
from optim_extra import alpha_max
from source_extra import cv_one_se, nested_one_se
from extra_utils import finish_extra
from plot_utils import savefig

X, y, _ = marketing_data(n=120, p=8, seed=620, noise=2.)
X, y = np.array(X), np.array(y)
# Это описательный путь; CV ниже получает исходные X и y.
Z = StandardScaler().fit_transform(X)
rho = .7
cutoff = alpha_max(Z, y, l1_ratio=rho)
alphas = cutoff * np.geomspace(1.01, .002, 16)
warm, cold, weights = [], [], []
w0 = None
max_difference = 0.
for alpha in alphas:
    settings = dict(alpha=float(alpha), l1_ratio=rho,
                    max_iter=30000, tol=1e-9)
    a = coordinate_descent(Z, y, w0=w0, **settings)
    b = coordinate_descent(Z, y, **settings)
    assert a["converged"] and b["converged"]
    w0 = a["coef"]
    warm.append(a["n_iter"])
    cold.append(b["n_iter"])
    weights.append(w0)
    max_difference = max(max_difference, float(np.max(
        np.abs(np.array(w0) - b["coef"]))))
assert max_difference < 1e-6
grid = np.geomspace(.005, 1., 16)
cv = cv_one_se(X, y, grid, rho)
nested = nested_one_se(X, y, grid, rho)
# Антипример: scaler видит все строки до внутренней CV модели.
leaky = make_pipeline(StandardScaler(), ElasticNetCV(
    alphas=grid.tolist(), l1_ratio=rho, tol=1e-9,
    cv=KFold(5, shuffle=True, random_state=620), max_iter=30000))
leaky.fit(X, y)
leaky_seen = int(leaky.named_steps["standardscaler"].n_samples_seen_)
assert leaky_seen == 120
assert cv["scaler_training_counts"] == [96] * 5
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4))
weights = np.array(weights)
for j in [0, 1, 4]:
    axes[0].semilogx(alphas, weights[:, j], label=f"x{j+1}")
axes[0].set(xlabel="Общая сила α", ylabel="Коэффициент")
axes[0].legend()
axes[1].semilogx(alphas, cold, "o-", label="Холодный старт")
axes[1].semilogx(alphas, warm, "o-", label="Тёплый старт")
axes[1].set(xlabel="Общая сила α", ylabel="Проходы нашего CD")
axes[1].legend()
savefig(fig, "e23_warm_path")
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.errorbar(grid, cv["mean"], yerr=cv["se"], fmt="o-",
             capsize=3, label="Средняя MSE ± SE")
ax.axhline(cv["threshold"], color=".4", ls=":", label="Порог 1SE")
ax.axvline(cv["best_alpha"], ls="--", label="Минимум MSE")
ax.axvline(cv["one_se_alpha"], color="#CE7334", ls="--",
            label="Выбор 1SE")
ax.set(xscale="log", xlabel="Общая сила α", ylabel="MSE на фолдах")
ax.legend(loc="upper left", fontsize=11)
savefig(fig, "e23_one_se")
RESULT = finish_extra("e23", {
    "alpha_max": cutoff, "alphas": alphas.tolist(),
    "warm_iterations": warm, "cold_iterations": cold,
    "warm_total": sum(warm), "cold_total": sum(cold),
    "max_coef_difference": max_difference, "cv": cv,
    "nested": nested,
    "nested_mean_mse": float(np.mean([r["mse"] for r in nested])),
    "scaler_before_internal_cv_seen": leaky_seen,
    "unequal_fold_mean": float(np.mean([1., 9.])),
    "unequal_pooled_mean": float(np.average([1., 9.],
                                             weights=[2, 1])),
})
FIGURES = ["e23_warm_path", "e23_one_se"]
