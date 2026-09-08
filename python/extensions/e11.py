"""Д11. Точный одномерный MAP; расчёты стандартной библиотекой."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extra_utils import finish_extra
from plot_utils import plt, savefig
from stat_extra import shifted_laplace_map, trapezoid

FIGURES = ["e11_posterior", "e11_prior_changes"]
n, sigma2, tau, c = 20, 4.0, 0.4, 1.0
# Детерминированные данные: XᵀX/n = 1, Xᵀy/n = c, b = 0.
X = [-1.0, 1.0] * (n // 2)
y = [v * c for v in X]
alpha = sigma2 / (n * tau)
grid = [-4 + i / 1000 for i in range(8001)]
loglik = [-n * (w - c) ** 2 / (2 * sigma2) for w in grid]
logprior = [-abs(w) / tau - math.log(2 * tau) for w in grid]
logpost = [a + b for a, b in zip(loglik, logprior)]


def density_on_grid(logvalues):
    top = max(logvalues)
    values = [math.exp(v - top) for v in logvalues]
    area = trapezoid(grid, values)
    return [v / area for v in values]


posterior = density_on_grid(logpost)
map_w = shifted_laplace_map(c, 1, alpha)
post_mean = trapezoid(grid, [w * v for w, v in zip(grid, posterior)])
fig, ax = plt.subplots(figsize=(7.6, 4.4))
for values, label in [(loglik, "Likelihood / площадь"),
                      (logprior, "Prior"), (logpost, "Posterior")]:
    ax.plot(grid, density_on_grid(values), label=label)
for value, label, color in [(c, "MLE", "#5F6368"),
                             (map_w, "MAP", "#8E598C"),
                             (post_mean, "Среднее", "#469278")]:
    ax.axvline(value, ls="--", color=color, label=label)
ax.set(xlim=(-1, 2.2), xlabel="Коэффициент w", ylabel="Высота")
ax.legend(ncol=2)
savefig(fig, FIGURES[0])

fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.3))
sizes = [10, 20, 40, 80]
for prior_scale in [0.2, 0.4, 0.8]:
    penalties = [sigma2 / (size * prior_scale) for size in sizes]
    axes[0].plot(sizes, penalties, "o-", label=f"τ = {prior_scale}")
centers = [-2 + i / 50 for i in range(201)]
for prior_mu in [0, 1.2]:
    values = [shifted_laplace_map(z, 1, alpha, prior_mu)
              for z in centers]
    axes[1].plot(centers, values, label=f"Центр prior {prior_mu}")
axes[1].plot(centers, centers, ":", color="black", label="MLE")
axes[0].set(xlabel="Число наблюдений n", ylabel="Штраф α")
axes[1].set(xlabel="Максимум likelihood c", ylabel="MAP")
for ax in axes:
    ax.legend()
savefig(fig, FIGURES[1])

# Сравниваем разности целей, чтобы убрать только константы по w.
differences = []
for w in [-1, 0, 0.5, 1, 2]:
    H = n * (w - c) ** 2 / (2 * sigma2) + abs(w) / tau
    H0 = n * c ** 2 / (2 * sigma2)
    J = 0.5 * (w - c) ** 2 + alpha * abs(w)
    differences.append(abs((H - H0) * sigma2 / n - J + c ** 2 / 2))
unequal = [shifted_laplace_map(z, 1, sigma2 / (n * t), m)
           for z, t, m in [(1, 0.4, 0), (0.4, 2, -0.5)]]
RESULT = {"n": n, "sigma2": sigma2, "tau": tau, "alpha": alpha,
          "mle": c, "map": map_w, "posterior_mean_grid": post_mean,
          "grid_step": 0.001, "unequal_shifted_map": unequal,
          "objective_scaling_error": max(differences)}
assert max(differences) < 1e-12
finish_extra("e11", RESULT)
