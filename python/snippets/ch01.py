"""Глава 01. Сначала предсказания и видимые остатки."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from pure_core import predict, residuals, mae, mse
from plot_utils import savefig
from experiments_a import finish

X, y = [[1], [2], [3]], [3, 5, 8]
w, b = [2], 1
pred = predict(X, w, b)
res = residuals(y, pred)
cases = [("Исходная", [2], 1), ("Изменили w", [2.5], 1),
         ("Изменили b", [2], 2)]
RESULT = {"X": X, "y": y, "coef": w, "intercept": b,
          "prediction": pred, "residual": res,
          "mae": mae(y, pred), "mse": mse(y, pred), "cases": []}
FIGURES = ["f01_residuals", "f01_predictions"]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot([.7, 3.3], [2 * .7 + b, 2 * 3.3 + b], label="ŷ = 2x + 1")
ax.scatter([1, 2, 3], y, s=65, zorder=4, label="Наблюдения y")
ax.vlines([1, 2, 3], pred, y, color="#D55E00", lw=4,
          label="Остаток r = y − ŷ")
ax.annotate("r₃ = 8 − 7 = 1", (3, 7.5), xytext=(1.8, 8.4),
            arrowprops={"arrowstyle": "->"})
ax.set(xlabel="Признак x", ylabel="Отклик / прогноз", ylim=(2, 9))
ax.legend(loc="upper left")
savefig(fig, FIGURES[0])

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for label, coef, intercept in cases:
    prediction = predict(X, coef, intercept)
    error = mae(y, prediction)
    RESULT["cases"].append({"name": label, "coef": coef,
                            "intercept": intercept,
                            "prediction": prediction, "mae": error})
    axes[0].plot([1, 2, 3], prediction, marker="o", label=label)
axes[0].scatter([1, 2, 3], y, marker="x", c="black", s=80,
                label="Наблюдения")
axes[0].set(xlabel="Признак x", ylabel="Прогноз", title="Меняем параметры")
axes[0].legend(fontsize=11)
axes[1].bar([row[0] for row in cases],
            [row["mae"] for row in RESULT["cases"]], color="#0072B2")
axes[1].set(ylabel="Средняя абсолютная ошибка", title="Итог каждого выбора")
axes[1].tick_params(axis="x", rotation=15)
savefig(fig, FIGURES[1])
finish(1, RESULT)

