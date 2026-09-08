"""Д09. Мост к NumPy: независимые обучения и bias--variance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from extra_utils import finish_extra
from plot_utils import plt, savefig
from pure_core import marketing_data

FIGURES = ["e09_bias_variance", "e09_coef_prediction",
           "e09_mean_median"]
R, n, p, sigma = 240, 45, 30, 4.0
penalties = [0.0, 0.01, 0.1, 1.0, 10.0]
Xe, truth, _ = marketing_data(320, p, 97000, 0, 0.99)
Xe, truth = np.asarray(Xe), np.asarray(truth)
pred = np.empty((len(penalties), R, len(truth)))
coef = np.empty((len(penalties), R, p))
probe_pred = np.empty((len(penalties), R))
response_error = np.zeros(len(penalties))
probe = np.zeros(p)
probe[:3], probe[4] = [12, 5, 3], 0.99
noise_rng = np.random.default_rng(907)
for rep in range(R):
    X, y, _ = marketing_data(n, p, 7000 + rep, sigma, 0.99)
    X, y = np.asarray(X), np.asarray(y)
    # Масштабирование заново только по текущему обучению.
    xm, ym, scale = X.mean(0), y.mean(), X.std(0)
    Z, yc = (X - xm) / scale, y - ym
    gram, rhs = Z.T @ Z / n, Z.T @ yc / n
    new_y = truth + noise_rng.normal(0, sigma, len(truth))
    for k, lam in enumerate(penalties):
        if lam == 0:
            ws = np.linalg.lstsq(Z, yc, rcond=None)[0]
        else:
            ws = np.linalg.solve(gram + lam * np.eye(p), rhs)
        w = ws / scale
        b = ym - xm @ w
        coef[k, rep], pred[k, rep] = w, Xe @ w + b
        probe_pred[k, rep] = probe @ w + b
        response_error[k] += np.mean((pred[k, rep] - new_y) ** 2) / R

average = pred.mean(axis=1)
bias2 = np.mean((average - truth) ** 2, axis=1)
variance = np.mean(np.var(pred, axis=1, ddof=0), axis=1)
noiseless = np.mean((pred - truth) ** 2, axis=(1, 2))
assert np.max(abs(noiseless - bias2 - variance)) < 1e-10
expected = bias2 + variance + sigma ** 2
fig, ax = plt.subplots(figsize=(7.6, 4.2))
for values, label in [(bias2, "Квадрат смещения"),
                      (variance, "Разброс прогноза"),
                      (expected, "Сумма с шумом 16")]:
    ax.plot(penalties, values, "o-", label=label)
ax.axhline(sigma ** 2, color="#5F6368", ls=":", label="Шум")
ax.set(xscale="symlog", xlabel="Штраф Ridge λ₂", ylabel="MSE")
ax.set_xscale("symlog", linthresh=0.01)
ax.set_xlim(0, 12)
ax.set_xticks(penalties, ["0", "0,01", "0,1", "1", "10"])
ax.legend()
savefig(fig, FIGURES[0])

fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.3))
for k, label in [(0, "МНК"), (2, "Ridge λ₂ = 0,1")]:
    axes[0].scatter(coef[k, :, 0], coef[k, :, 4], s=13,
                    alpha=0.5, label=label)
    axes[1].hist(probe_pred[k], bins=20, density=True,
                 histtype="step", linewidth=2, label=label)
axes[0].scatter([3], [0], marker="x", s=80, color="black")
axes[0].set(xlabel="Коэффициент w₁", ylabel="Коэффициент w₅")
axes[1].axvline(49, ls="--", color="black")
axes[1].set(xlabel="Прогноз в одной точке", ylabel="Плотность")
for ax in axes:
    ax.legend()
savefig(fig, FIGURES[1])
# Константная регрессия: МНК = среднее, LAD = медиана.
small_rng = np.random.default_rng(917)
centers = {}
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), sharey=True)
for ax, label in zip(axes, ["Нормальный шум", "Шум Лапласа"]):
    draws = (small_rng.normal(size=(2000, 41)) if label[0] == "Н"
             else small_rng.laplace(scale=1 / np.sqrt(2),
                                    size=(2000, 41)))
    avg, med = draws.mean(axis=1), np.median(draws, axis=1)
    centers[label] = [float(np.mean(avg ** 2)),
                      float(np.mean(med ** 2))]
    for values, name in [(avg, "МНК: среднее"), (med, "LAD: медиана")]:
        ax.hist(values, bins=30, density=True, histtype="step",
                linewidth=2, label=name)
    ax.set(title=label, xlabel="Оценка центра", ylabel="Плотность")
    ax.legend()
savefig(fig, FIGURES[2])
# Локально EN аффинен по корреляциям при неизменных знаках.
G = np.array([[1.0, 0.8], [0.8, 1.0]])
corr = np.array([1.2, 0.9])
lambda1, lambda2 = 0.1, 0.2
inverse = np.linalg.inv(G + lambda2 * np.eye(2))
active_w = inverse @ (corr - lambda1)
delta_corr = np.array([0.0, 0.01])
changed_w = inverse @ (corr + delta_corr - lambda1)
assert np.all(active_w > 0) and np.all(changed_w > 0)
assert np.max(abs(changed_w - active_w
                  - inverse @ delta_corr)) < 1e-12
RESULT = {"R": R, "n": n, "p": p, "sigma": sigma,
          "penalties": penalties, "bias2": bias2,
          "variance": variance, "noise_plus_components": expected,
          "fresh_response_mse": response_error,
          "coef_sd_w1": coef[:, :, 0].std(axis=1),
          "coef_sd_w5": coef[:, :, 4].std(axis=1),
          "joint_effect_sd": (2 * coef[:, :, 0]
                              + 0.99 * coef[:, :, 4]).std(axis=1),
          "probe_mean": probe_pred.mean(axis=1),
          "probe_sd": probe_pred.std(axis=1), "center_mse": centers,
          "active_affine": {"before": active_w, "after": changed_w},
          "decomposition_error": max(abs(noiseless - bias2 - variance))}
finish_extra("e09", RESULT)
