"""Глава 09. Lasso: ошибка данных и плата за веса по отдельности."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from pure_core import marketing_data, scale_fit, scale_transform
from pure_core import lasso_cd, predict, mse
from plot_utils import savefig
from experiments_a import finish
from sklearn.model_selection import train_test_split

X, y, beta = marketing_data()
train, test = train_test_split(list(range(160)), test_size=.25,
                               random_state=42)
stats = scale_fit([X[i] for i in train])
Z = scale_transform([X[i] for i in train], stats)
target = [y[i] for i in train]
alphas = [0, .01, .03, .1, .3, 1, 3, 10]
records = []
for alpha in alphas:
    fit = lasso_cd(Z, target, alpha=alpha, max_iter=20000, tol=1e-8)
    error = mse(target, predict(Z, fit["coef"], fit["intercept"]))
    norm = sum(abs(w) for w in fit["coef"])
    records.append({"alpha": alpha, "coef": fit["coef"],
                    "intercept": fit["intercept"], "train_mse": error,
                    "half_mse": error / 2, "l1_norm": norm,
                    "penalty": alpha * norm,
                    "zeros": sum(abs(w) < 1e-10 for w in fit["coef"]),
                    "converged": fit["converged"],
                    "optimality_error": fit["optimality_error"]})
assert all(row["converged"] for row in records)
RESULT = {"seed": 42, "train_n": 120, "records": records,
          "feature_names": ["Реклама", "Цена", "Рассылки", "Шум 4",
                            "Коррелят рекламы", "Шум 6", "Шум 7",
                            "Шум 8"]}
FIGURES = ["f09_lasso_path", "f09_tradeoff"]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for j, name in enumerate(RESULT["feature_names"]):
    axes[0].plot(alphas, [r["coef"][j] for r in records], marker=".",
                 label=name, color=plt.get_cmap("tab10")(j))
axes[0].set(ylabel="Коэффициент после стандартизации",
            title="Коэффициенты Lasso", ylim=(-2.4, 10.1))
axes[0].legend(fontsize=11, ncol=2, loc="upper right")
axes[1].scatter(alphas, [r["zeros"] for r in records], s=42)
axes[1].set(ylabel="Число нулевых коэффициентов", ylim=(-.3, 8.5),
            yticks=range(9), title="Нули при проверенных alpha")
for ax in axes:
    ax.set_xscale("symlog", linthresh=.01)
    ax.set_xlim(-.001, 12)
    ax.set_xlabel("alpha")
savefig(fig, FIGURES[0])

fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
axes[0].plot(alphas, [r["half_mse"] for r in records], "o-")
axes[0].set(ylabel="MSE / 2 на обучении", title="Потеря на данных")
axes[1].plot(alphas, [r["penalty"] for r in records], "o-",
              color="#D55E00", label="alpha · сумма |w|")
axes[1].set(ylabel="Значение штрафа", title="Плата за выбранные веса")
axes[1].legend(fontsize=11)
for ax in axes:
    ax.set_xscale("symlog", linthresh=.01)
    ax.set_xlabel("alpha")
savefig(fig, FIGURES[1])
finish(9, RESULT)
