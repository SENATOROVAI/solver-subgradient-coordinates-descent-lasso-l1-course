"""Д18. Нормировка и MAP: явно используем SciPy для erfcx/quad."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy.integrate import quad
from extra_utils import finish_extra
from plot_utils import plt, savefig
from pure_core import soft_threshold
from extension_listings.en_prior_bridge import log_z, prior_pdf

FIGURES = ["e18_prior_product", "e18_penalty_shape"]
grid = [-4 + i / 100 for i in range(801)]
lap = [prior_pdf(t, 1, 0) for t in grid]
gauss = [prior_pdf(t, 0, 1) for t in grid]
elastic = [prior_pdf(t, 1, 1) for t in grid]
mixture = [(a + b) / 2 for a, b in zip(lap, gauss)]
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.3))
for ax in axes:
    for values, label in [(lap, "Лапласа"), (gauss, "Гаусса"),
                          (elastic, "Произведение: EN"),
                          (mixture, "Смесь 50/50")]:
        ax.plot(grid, values, label=label)
    ax.set(xlabel="Коэффициент t", ylabel="Плотность")
axes[0].set_xlim(-2, 2)
axes[1].set(yscale="log", ylim=(1e-6, 1))
axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
savefig(fig, FIGURES[0])

fig, ax = plt.subplots(figsize=(7.6, 4.2))
for values, label in [(elastic, "EN: |t| + t²/2"),
                      (mixture, "Смесь: −log p(t) + log p(0)")]:
    middle = values[len(grid) // 2]
    ax.plot(grid, [-math.log(v / middle) for v in values], label=label)
ax.set(xlabel="Коэффициент t", ylabel="Рост отрицательного log prior")
ax.legend()
savefig(fig, FIGURES[1])

integrals = []
for a, d in [(1, 0), (0, 1), (1, 1), (2, 0.5), (100, 1)]:
    value, error = quad(lambda t: 2 * prior_pdf(t, a, d), 0, math.inf)
    integrals.append({"a": a, "d": d, "Z": math.exp(log_z(a, d)),
                      "integral": value, "quad_error": error})
    assert abs(value - 1) < 1e-8
try:
    direct = math.exp(5000) * math.erfc(100 / math.sqrt(2))
    direct_status = str(direct)
except OverflowError:
    direct_status = "OverflowError"

n, sigma2, a, d, center = 100, 4.0, 2.0, 0.5, 1.4
l1, l2 = sigma2 * a / n, sigma2 * d / n
map_w = soft_threshold(center, l1) / (1 + l2)
# Моменты и производные log Z: две независимые проверки.
abs_mean = quad(lambda t: 2 * t * prior_pdf(t, 1, 1), 0, math.inf)[0]
variance = quad(lambda t: 2 * t * t * prior_pdf(t, 1, 1),
                0, math.inf)[0]
h = 1e-5
from_a = -(log_z(1 + h, 1) - log_z(1 - h, 1)) / (2 * h)
from_d = -(log_z(1, 1 + h) - log_z(1, 1 - h)) / h
assert abs(abs_mean - from_a) < 1e-8
assert abs(variance - from_d) < 1e-8
RESULT = {"integrals": integrals, "direct_large_a": direct_status,
          "lambda1": l1, "lambda2": l2, "alpha": l1 + l2,
          "l1_ratio": l1 / (l1 + l2), "map": map_w,
          "abs_mean_11": abs_mean, "variance_11": variance,
          "moment_derivative_errors": [abs(abs_mean - from_a),
                                       abs(variance - from_d)]}
finish_extra("e18", RESULT)
