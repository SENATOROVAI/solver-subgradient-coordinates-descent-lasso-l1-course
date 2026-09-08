"""Глава 03. Измерить качество и выбрать параметры -- разные шаги."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from pure_core import predict, mae, mse, grid_mae
from plot_utils import savefig
from experiments_a import finish

X, y = [[1], [2], [3]], [3, 5, 8]
candidates = [([2], 0), ([2], 1), ([3], 0)]
scores = [mae(y, predict(X, w, b)) for w, b in candidates]
learned = grid_mae(X, y, [1 + i / 4 for i in range(9)],
                   [-1 + i / 4 for i in range(13)])
zero_y = [0, 0, 0, 0]
pred_a, pred_b = [0, 0, 0, 4], [1.2] * 4
metric_scores = {name: {"mae": mae(zero_y, pred),
                         "mse": mse(zero_y, pred)}
                 for name, pred in [("A", pred_a), ("B", pred_b)]}
RESULT = {"fixed_scores": scores,
          "learned": {k: learned[k] for k in
                      ["coef", "intercept", "loss"]},
          "metric_y": zero_y, "prediction_A": pred_a,
          "prediction_B": pred_b, "metric_scores": metric_scores}
FIGURES = ["f03_grid_choice", "f03_metric_choice"]

fig, ax = plt.subplots(figsize=(8, 4.4))
values = scores + [learned["loss"]]
bars = ax.bar(["w=2, b=0", "w=2, b=1", "w=3, b=0",
               "Лучший на сетке"], values,
              color=["#999999"] * 3 + ["#009E73"])
ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=5)
ax.set(ylabel="MAE на трех наблюдениях", ylim=(0, 1.55),
       title="Метрика оценивает; перебор выбирает параметры")
savefig(fig, FIGURES[0])

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, metric in zip(axes, ["mae", "mse"]):
    values = [metric_scores[name][metric] for name in ["A", "B"]]
    bars = ax.bar(["Прогноз A", "Прогноз B"], values,
                  color=["#0072B2", "#D55E00"])
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=5)
    ax.set(ylabel=metric.upper(), ylim=(0, max(values) * 1.25),
           title="Лучше " + ("A" if metric == "mae" else "B"))
fig.suptitle("Те же прогнозы, но другой порядок качества")
savefig(fig, FIGURES[1])
finish(3, RESULT)

