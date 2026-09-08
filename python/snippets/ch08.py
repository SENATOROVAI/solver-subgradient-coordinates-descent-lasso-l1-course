"""Глава 08. Единицы измерения и статистики только по train."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from pure_core import marketing_data, predict, scale_fit, scale_transform
from pure_core import unscale_coefficients
from plot_utils import savefig
from experiments_a import finish
from sklearn.model_selection import train_test_split

X, y, beta = marketing_data()
X = [row + [7.] for row in X]
beta = beta + [0.]
train, test = train_test_split(list(range(160)), test_size=.25,
                               random_state=42)
stats = scale_fit([X[i] for i in train])
Z = scale_transform(X, stats)
scaled_w = [w * s for w, s in zip(beta, stats["scale"])]
scaled_b = 20 + sum(m * w for m, w in zip(stats["mean"], beta))
back_w, back_b = unscale_coefficients(scaled_w, scaled_b, stats)
pred_raw = predict(X, beta, 20)
pred_scaled = predict(Z, scaled_w, scaled_b)
X_units = [[row[0] * 1000] + row[1:] for row in X]
w_units = [beta[0] / 1000] + beta[1:]
pred_units = predict(X_units, w_units, 20)
scale_gap = max(abs(a - b) for a, b in zip(pred_raw, pred_scaled))
unit_gap = max(abs(a - b) for a, b in zip(pred_raw, pred_units))
assert scale_gap < 1e-10 and unit_gap < 1e-10
assert all(abs(Z[i][-1]) < 1e-12 for i in range(160))
RESULT = {"train_n": len(train), "test_n": len(test), "stats": stats,
          "raw_coef": beta, "raw_intercept": 20,
          "scaled_coef": scaled_w, "scaled_intercept": scaled_b,
          "restored_coef": back_w, "restored_intercept": back_b,
          "converted_coef": w_units, "scaling_prediction_gap": scale_gap,
          "unit_prediction_gap": unit_gap, "constant_value": 7,
          "constant_scale": stats["scale"][-1]}
FIGURES = ["f08_scaling", "f08_coefficients"]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
names = ["Реклама", "Цена", "Рассылки", "Константа"]
cols = [0, 1, 2, 8]
for ax, rows, title in [(axes[0], X, "Исходные единицы"),
                         (axes[1], Z, "После стандартизации train")]:
    ax.boxplot([[rows[i][j] for i in train] for j in cols],
                tick_labels=names, showfliers=False)
    ax.set(ylabel="Значение признака", title=title)
    ax.tick_params(axis="x", rotation=15)
savefig(fig, FIGURES[0])

fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
axes[0].bar([.85, 1.85, 2.85], beta[:3], width=.3,
             label="Исходные единицы")
axes[0].bar([1.15, 2.15, 3.15], scaled_w[:3], width=.3,
             label="После стандартизации")
axes[0].set(xticks=[1, 2, 3], xticklabels=names[:3],
            ylabel="Коэффициент", title="Веса меняются вместе с единицами")
axes[0].legend(fontsize=11)
axes[1].scatter(pred_raw, pred_scaled, s=17, label="Стандартизация")
axes[1].plot([min(pred_raw), max(pred_raw)],
              [min(pred_raw), max(pred_raw)], c="black", ls="--")
axes[1].set(xlabel="Прогноз в исходных единицах",
            ylabel="Прогноз после преобразования",
            title="Предсказания совпадают")
savefig(fig, FIGURES[1])
finish(8, RESULT)
