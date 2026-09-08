"""Д07. PDF/CDF, Q-Q и совместная MLE центра и масштаба."""
import math
import random
import sys
from pathlib import Path
from statistics import NormalDist, mean
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extra_utils import finish_extra
from plot_utils import plt, savefig
from stat_extra import (laplace_cdf, laplace_pdf, laplace_ppf,
                        location_scale_mle, normal_cdf, normal_pdf)

FIGURES = ["e07_pdf_cdf", "e07_qq", "e07_residual_scale"]
rng = random.Random(905)
s = 1 / math.sqrt(2)
grid = [-4 + i / 50 for i in range(401)]
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
for label, pdf, cdf, scale in [
        ("Нормальное", normal_pdf, normal_cdf, 1),
        ("Лапласа", laplace_pdf, laplace_cdf, s)]:
    axes[0].plot(grid, [pdf(x, 0, scale) for x in grid], label=label)
    axes[1].plot(grid, [cdf(x, 0, scale) for x in grid], label=label)
for ax, title in zip(axes, ["Плотность PDF", "Вероятность CDF"]):
    ax.set(xlabel="Значение z", ylabel=title)
    ax.legend()
savefig(fig, FIGURES[0])

# Это новая выборка шума, не остатки после подгонки регрессии.
sample = [laplace_ppf(rng.random(), 0, s) for _ in range(240)]
fit = location_scale_mle(sample)
ordered = sorted(sample)
probs = [(i + 0.5) / len(sample) for i in range(len(sample))]
normal = NormalDist(fit["normal_mu"], fit["normal_sigma"])
quantiles = [
    [normal.inv_cdf(q) for q in probs],
    [laplace_ppf(q, fit["laplace_mu"], fit["laplace_scale"])
     for q in probs]]
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
errors = []
for ax, q, title in zip(axes, quantiles, ["Нормальное", "Лапласа"]):
    ax.scatter(q, ordered, s=13, alpha=0.65)
    low, high = min(q + ordered), max(q + ordered)
    ax.plot([low, high], [low, high], "--", color="#CE7334")
    ax.set(title=title, xlabel="Квантили модели",
           ylabel="Квантили выборки")
    errors.append(math.sqrt(mean([(a - b) ** 2
                                  for a, b in zip(q, ordered)])))
savefig(fig, FIGURES[1])
# Одинаковые стандартизованные шумы, разные условные масштабы.
x = [4 * i / 239 for i in range(240)]
base_noise = [laplace_ppf(rng.random()) for _ in x]
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), sharey=True)
scale_reports = []
for ax, varied in zip(axes, [False, True]):
    scales = [0.3 + 0.225 * v if varied else 0.3 for v in x]
    y = [1 + 2 * v + s0 * u
         for v, s0, u in zip(x, scales, base_noise)]
    xm, ym = mean(x), mean(y)
    slope = sum((v - xm) * (z - ym) for v, z in zip(x, y))
    slope /= sum((v - xm) ** 2 for v in x)
    fitted = [ym + slope * (v - xm) for v in x]
    residual = [z - zhat for z, zhat in zip(y, fitted)]
    ax.scatter(fitted, residual, s=13, alpha=0.6)
    ax.axhline(0, color="black", ls="--")
    ax.set(xlabel="Подогнанный ответ", ylabel="Остаток y − ŷ",
           title="Растущий масштаб" if varied else "Постоянный масштаб")
    scale_reports.append([mean([abs(v) for v in residual[:60]]),
                          mean([abs(v) for v in residual[-60:]])])
savefig(fig, FIGURES[2])
RESULT = {"n": len(sample), "seed": 905, "fit": fit,
          "manual": location_scale_mle([1, 2, 6]),
          "normal_inside_one_sd": normal_cdf(1) - normal_cdf(-1),
          "laplace_inside_one_sd": laplace_cdf(1, 0, s)
          - laplace_cdf(-1, 0, s), "qq_rmse": errors,
          "laplace_peak_scale_point1": laplace_pdf(0, 0, 0.1),
          "residual_mean_abs_first_last": scale_reports,
          "product_2000_pdf_at_one": laplace_pdf(1) ** 2000,
          "loglik_2000_at_one": 2000 * (-math.log(2) - 1)}
assert RESULT["laplace_peak_scale_point1"] == 5
assert abs(RESULT["manual"]["laplace_scale"] - 5 / 3) < 1e-12
finish_extra("e07", RESULT)
