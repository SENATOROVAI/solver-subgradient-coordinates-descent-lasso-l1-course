"""Д02. Взвешенный постоянный прогноз и накопленная масса."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from mae_extra import weighted_mae, quantile_interval
from plot_utils import savefig
from extra_utils import finish_extra

FIGURES = ["e02_weighted_loss", "e02_weighted_cdf"]
y = [1, 2, 4, 10]
weights = [1, 1, 1, 7]
grid = [k / 20 for k in range(241)]
ordinary = [weighted_mae(y, [b] * 4) for b in grid]
weighted = [weighted_mae(y, [b] * 4, weights) for b in grid]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.plot(grid, ordinary, label="Все веса равны 1")
ax.plot(grid, weighted, label="Веса 1, 1, 1, 7")
ax.axvspan(2, 4, alpha=0.12, color="#24577C")
ax.scatter([3, 10], [2.75, 2.3], s=65, zorder=4)
ax.set(xlabel="Постоянный прогноз b", ylabel="Средняя абсолютная ошибка")
ax.legend()
savefig(fig, FIGURES[0])
fig, ax = plt.subplots(figsize=(7.6, 4.2))
for a, label in [([1] * 4, "Обычная доля"),
                 (weights, "Взвешенная доля")]:
    cdf = [sum(w for v, w in zip(y, a) if v <= t) / sum(a)
           for t in grid]
    ax.step(grid, cdf, where="post", label=label)
ax.axhline(0.5, color="#5F6368", ls="--", label="Половина массы")
ax.set(xlabel="Порог t", ylabel="Масса ответов не выше t", ylim=(0, 1.06))
ax.legend()
savefig(fig, FIGURES[1])
replicated = [v for v, a in zip(y, weights) for _ in range(a)]
assert quantile_interval(y) == (2.0, 4.0)
assert quantile_interval(y, weights=weights) == (10.0, 10.0)
assert quantile_interval([0, 2, 99, 10], weights=[1, 1, 0, 2]) == (2, 10)
for b in [0, 3, 10]:
    value = weighted_mae(y, [b] * 4, weights)
    assert abs(value - weighted_mae(y, [b] * 4,
                                   [10 * a for a in weights])) < 1e-12
    assert abs(value - weighted_mae(
        replicated, [b] * len(replicated))) < 1e-12
RESULT = {"ordinary_interval": quantile_interval(y),
          "weighted_interval": quantile_interval(y, weights=weights),
          "ordinary_minimum": min(ordinary),
          "weighted_minimum": min(weighted),
          "weight_replication_equal": True,
          "zero_weight_interval": [2, 10]}
finish_extra("e02", RESULT)
