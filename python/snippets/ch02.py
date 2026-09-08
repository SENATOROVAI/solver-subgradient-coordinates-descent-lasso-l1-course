"""Глава 02. Знаки ошибок и явный мост к библиотекам."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from pure_core import predict, residuals, mae
from plot_utils import savefig
from experiments_a import finish

X, y = [[1], [2], [3]], [3, 5, 8]
base = predict(X, [2], 1)
shifted = predict(X, [2], 4 / 3)
r = residuals(y, shifted)
signed_mean = sum(r) / len(r)
absolute_mean = mae(y, shifted)
# Мост NumPy / scikit-learn: проверка того же расчета.
import numpy as np
from sklearn.metrics import mean_absolute_error
array_prediction = np.asarray(X) @ np.asarray([2.]) + 4 / 3
numpy_mae = float(np.mean(np.abs(np.asarray(y) - array_prediction)))
library_mae = float(mean_absolute_error(y, array_prediction))
assert abs(numpy_mae - absolute_mean) < 1e-12
assert abs(library_mae - absolute_mean) < 1e-12
RESULT = {"base_prediction": base, "base_mae": mae(y, base),
          "shifted_prediction": shifted, "residual": r,
          "signed_mean": signed_mean, "mae": absolute_mean,
          "numpy_mae": numpy_mae, "sklearn_mae": library_mae}
FIGURES = ["f02_absolute", "f02_mean_error"]

fig, ax = plt.subplots(figsize=(8, 4.3))
ax.bar([.82, 1.82, 2.82], r, width=.34, label="r = y − ŷ")
ax.bar([1.18, 2.18, 3.18], [abs(v) for v in r], width=.34,
       label="|r|", color="#D55E00")
ax.axhline(0, color="black", lw=.8)
ax.set(xticks=[1, 2, 3], xlabel="Номер наблюдения",
       ylabel="Ошибка", title="При w = 2 и b = 4/3 знаки сокращаются")
ax.legend()
savefig(fig, FIGURES[0])

fig, ax = plt.subplots(figsize=(8, 4.3))
values = [signed_mean, absolute_mean, numpy_mae, library_mae]
bars = ax.bar(["Среднее r", "MAE: списки", "MAE: NumPy",
               "MAE: sklearn"], values,
              color=["#999999", "#0072B2", "#56B4E9", "#009E73"])
ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=5)
ax.set(ylabel="Средняя ошибка", ylim=(-.04, .56),
       title="Нулевое среднее r не означает точного прогноза")
savefig(fig, FIGURES[1])
finish(2, RESULT)

