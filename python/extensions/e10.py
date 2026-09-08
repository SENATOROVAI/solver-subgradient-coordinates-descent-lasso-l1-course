"""Д10. Повторное обучение после изменения единиц признака."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pure_core import marketing_data
from plot_utils import savefig
from extra_utils import finish_extra

X, y, truth = marketing_data(n=120, seed=824)
X, y = np.array(X), np.array(y)
changed = X.copy()
changed[:, 0] *= 1000
raw = Lasso(alpha=.3, tol=1e-10, max_iter=20000).fit(X, y)
raw_new = Lasso(alpha=.3, tol=1e-10, max_iter=20000).fit(changed, y)

def fit_scaled(data, target):
    return make_pipeline(StandardScaler(), Lasso(
        alpha=.3, tol=1e-10, max_iter=20000)).fit(data, target)

scaled = fit_scaled(X, y)
scaled_new = fit_scaled(changed, y)
p0, p1 = scaled.predict(X), scaled_new.predict(changed)
assert np.max(abs(p0 - p1)) < 1e-8
shift = fit_scaled(X, y + 100)
assert np.max(abs(shift.predict(X) - p0 - 100)) < 1e-8
fig, axes = plt.subplots(1, 2, figsize=(9., 4.2))
for ax, pred, new, label in [
        (axes[0], raw.predict(X), raw_new.predict(changed), 'Без scaler'),
        (axes[1], p0, p1, 'Scaler внутри Pipeline')]:
    ax.scatter(pred, new, s=17, alpha=.7)
    lo, hi = min(pred.min(), new.min()), max(pred.max(), new.max())
    ax.plot([lo, hi], [lo, hi], '--', color='black', label='Равенство')
    ax.set(xlabel='Прогноз: исходные единицы',
           ylabel='Прогноз: новые единицы', title=label)
    ax.legend()
savefig(fig, 'e10_refit_units')

# Редкая единица после стандартизации и форма распределения.
rare = np.array([0.] * 99 + [1.]).reshape(-1, 1)
z = StandardScaler().fit_transform(rare)[:, 0]
rng = np.random.default_rng(824)
skew = rng.exponential(size=(1000, 1))
zs = StandardScaler().fit_transform(skew)[:, 0]
fig, axes = plt.subplots(1, 2, figsize=(9., 4.2))
axes[0].bar(['Исходное 0', 'Исходное 1'], [z[0], z[-1]])
axes[0].set(ylabel='Стандартизованное значение', title='Единица в 1% строк')
axes[1].hist(zs, bins=30, density=True, label='После StandardScaler')
axes[1].set(xlabel='Стандартизованное значение', ylabel='Плотность частот',
            title='Асимметрия сохранилась')
axes[1].legend()
savefig(fig, 'e10_scaling_limits')
# Среднее центрирование не является обучением свободного члена LAD.
constant = np.ones((3, 1))
target = np.array([0., 0., 9.])
lad = QuantileRegressor(alpha=0., quantile=.5).fit(constant, target)
assert np.allclose(lad.predict(constant), 0.)
RESULT = {'raw_refit_prediction_change': float(np.max(abs(
            raw.predict(X) - raw_new.predict(changed)))),
          'scaled_refit_prediction_change': float(np.max(abs(p0 - p1))),
          'rare_standardized': [float(z[0]), float(z[-1])],
          'mean_target': float(target.mean()),
          'lad_prediction': lad.predict(constant).tolist(),
          'correlation_change': float(np.max(abs(
              np.corrcoef(X, rowvar=False) - np.corrcoef(
                  StandardScaler().fit_transform(X), rowvar=False))))}
FIGURES = ['e10_refit_units', 'e10_scaling_limits']
finish_extra('e10', RESULT)
