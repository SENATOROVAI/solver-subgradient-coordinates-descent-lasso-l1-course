"""Глава 04. Абсолютная ошибка константы и интервал медиан."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from pure_core import mae, median
from plot_utils import savefig
from experiments_a import finish

y = [1, 2, 4, 9]
constants = [i / 40 for i in range(401)]
loss = [mae(y, [c] * len(y)) for c in constants]
probes = [0, 1, 2, 3, 4, 6, 9]
RESULT = {"y": y, "median_convention": median(y),
          "minimizer_interval": [2, 4],
          "probes": [{"b": c, "mae": mae(y, [c] * 4)}
                     for c in probes]}
FIGURES = ["f04_absolute", "f04_median"]

fig, ax = plt.subplots(figsize=(8, 4.2))
r = [-3 + i / 50 for i in range(301)]
ax.plot(r, [abs(v) for v in r], lw=2.5, label="|r|")
ax.axvline(0, color="black", lw=.8)
ax.axhline(0, color="black", lw=.8)
ax.annotate("Слева наклон −1", (-1.7, 1.7), xytext=(-2.9, 3.15),
            arrowprops={"arrowstyle": "->"})
ax.annotate("Справа наклон +1", (1.5, 1.5), xytext=(.3, 3.15),
            arrowprops={"arrowstyle": "->"})
ax.set(xlabel="Остаток r = y − ŷ", ylabel="Абсолютная ошибка",
       ylim=(-.2, 3.6), title="В нуле нет одного обычного наклона")
savefig(fig, FIGURES[0])

fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
axes[0].plot(constants, loss, color="#0072B2")
axes[0].axvspan(2, 4, alpha=.2, color="#009E73",
                label="Все минимумы: [2, 4]")
axes[0].scatter(probes, [mae(y, [c] * 4) for c in probes], s=35)
axes[0].set(xlabel="Константный прогноз b", ylabel="MAE",
            title="Четное число наблюдений")
axes[0].legend(fontsize=11)
axes[1].scatter(range(1, 5), y, color="black", label="y = [1, 2, 4, 9]")
for c, style in [(2, "--"), (3, "-"), (4, ":")]:
    axes[1].axhline(c, ls=style, label=f"b = {c}: MAE = 2.5")
axes[1].set(xlabel="Номер наблюдения", ylabel="Наблюдение / прогноз",
            xticks=[1, 2, 3, 4], title="Три одинаково хороших константы")
axes[1].legend(fontsize=11)
savefig(fig, FIGURES[1])
finish(4, RESULT)
