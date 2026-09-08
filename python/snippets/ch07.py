"""Глава 07. Лишние признаки: обучение и отдельная валидация."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import random
import matplotlib.pyplot as plt
from pure_core import predict, mse
from plot_utils import savefig
from experiments_a import finish

rng = random.Random(42)
X = [[rng.gauss(0, 1) for _ in range(58)] for _ in range(240)]
beta = [3, -2, 1] + [0] * 55
y = [2 + sum(a * b for a, b in zip(row, beta)) + rng.gauss(0, 1)
     for row in X]
counts = [1, 2, 3, 5, 10, 20, 30, 40, 50, 58]
records, coefficients = [], []
# Явный мост NumPy: библиотечное решение OLS, не наш алгоритм.
import numpy as np
for p in counts:
    train_X = [row[:p] for row in X[:60]]
    val_X = [row[:p] for row in X[60:]]
    design = np.asarray([[1] + row for row in train_X])
    fitted = np.linalg.lstsq(design, np.asarray(y[:60]), rcond=None)[0]
    b, w = float(fitted[0]), fitted[1:].tolist()
    train_error = mse(y[:60], predict(train_X, w, b))
    val_error = mse(y[60:], predict(val_X, w, b))
    records.append({"p": p, "train_mse": train_error,
                    "validation_mse": val_error})
    coefficients.append(w)
best = min(records, key=lambda row: row["validation_mse"])
RESULT = {"seed": 42, "train_n": 60, "validation_n": 180,
          "noise_sd": 1, "beta": beta, "records": records,
          "selected_p": best["p"], "coef_p3": coefficients[2],
          "coef_p58": coefficients[-1]}
FIGURES = ["f07_overfit", "f07_noise"]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
for ax in axes:
    ax.plot(counts, [r["train_mse"] for r in records], "o-",
            label="Обучение: 60 строк")
    ax.plot(counts, [r["validation_mse"] for r in records], "o-",
            label="Валидация: 180 строк")
    ax.axvline(3, ls=":", c="black", label="3 истинных признака")
    ax.set(xlabel="Число признаков", ylabel="MSE")
axes[0].set(title="Вся шкала", yscale="log")
axes[1].set(title="Ближе: от 1 до 20 признаков", xlim=(0, 21),
            ylim=(0, 8))
axes[0].legend(fontsize=11)
savefig(fig, FIGURES[0])

fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
axes[0].bar([1, 2, 3], coefficients[2], width=.55,
             color="#0072B2", label="OLS: p = 3")
axes[0].scatter([1, 2, 3], beta[:3], color="black", s=55,
                marker="x", label="Истинные веса")
axes[0].set(xlabel="Номер признака", ylabel="Коэффициент",
            title="Три полезных признака", xticks=[1, 2, 3])
axes[0].legend(fontsize=11)
axes[1].bar(range(4, 59), coefficients[-1][3:], color="#D55E00")
axes[1].axhline(0, color="black", lw=.8)
axes[1].set(xlabel="Номер шумового признака", ylabel="Коэффициент",
            title="p = 58: истинные веса здесь равны нулю")
savefig(fig, FIGURES[1])
finish(7, RESULT)
