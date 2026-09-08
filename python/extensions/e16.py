"""Д16. Направления, обусловленность и проверяемые границы."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import numpy as np  # Явный мост: спектр, solve и расширенный lstsq.
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from pure_core import coordinate_descent, objective
from plot_utils import savefig
from extra_utils import finish_extra

x = np.array([-1., -1., 1., 1.])
z = np.array([-1., 1., -1., 1.])
corr = .99
X = np.column_stack([x, corr * x + math.sqrt(1 - corr ** 2) * z])
G = X.T @ X / len(X)
qminus = np.array([1., -1.]) / math.sqrt(2)
beta = np.array([math.sqrt(2), 0.])
y = X @ beta
c = X.T @ y / len(y)
lambda2 = .09
w = np.linalg.solve(G + lambda2 * np.eye(2), c)
perturbation = X @ qminus * .1
shifted = np.linalg.solve(G + lambda2 * np.eye(2),
                          X.T @ (y + perturbation) / len(y))
ols = np.linalg.solve(G, c)
ols_new = np.linalg.solve(G, X.T @ (y + perturbation) / len(y))
aug_X = np.vstack([X, math.sqrt(len(y) * lambda2) * np.eye(2)])
aug_y = np.r_[y, [0., 0.]]
w_aug = np.linalg.lstsq(aug_X, aug_y, rcond=None)[0]
ref = Ridge(alpha=len(y) * lambda2, fit_intercept=False).fit(X, y)
ours = coordinate_descent(X, y, lambda2, 0., tol=1e-12)
assert ours['converged']
agreement = max(float(np.max(np.abs(w - candidate)))
                for candidate in [w_aug, ref.coef_, ours['coef']])
assert agreement < 1e-9
values = np.geomspace(.0001, 10., 100)
eigen = np.linalg.eigvalsh(G)
condition = (eigen[-1] + values) / (eigen[0] + values)
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.5))
axes[0].loglog(values, condition)
axes[0].scatter([lambda2], [20.8], color='#CE7334', s=60)
axes[0].set(xlabel='Квадратичная сила λ₂', ylabel='Обусловленность',
            title='Стандартизация уже выполнена')
for direction, ev in [('Сумма: 1.99', eigen[1]),
                       ('Разность: 0.01', eigen[0])]:
    axes[1].semilogx(values, ev / (ev + values), label=direction)
axes[1].set(xlabel='Квадратичная сила λ₂', ylabel='Доля исходного сигнала')
axes[1].legend()
savefig(fig, 'e16_condition')

rng = np.random.default_rng(713)
raw = rng.normal(size=(15, 24))
centered = raw - raw.mean(axis=0)
Gram = centered.T @ centered / len(raw)
spectrum = np.linalg.eigvalsh(Gram)
shift_spectrum = np.linalg.eigvalsh(Gram + .2 * np.eye(24))
rank = int(np.linalg.matrix_rank(centered))
arbitrary = rng.normal(size=len(raw))
permuted = rng.permutation(arbitrary)
interpolation = []
for response in [arbitrary, permuted]:
    target = response - response.mean()
    solution = np.linalg.lstsq(centered, target, rcond=None)[0]
    residual_error = float(np.max(np.abs(centered @ solution - target)))
    interpolation.append(residual_error)
assert max(interpolation) < 1e-10
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.5))
axes[0].plot(spectrum, 'o-', label='Без L2')
axes[0].plot(shift_spectrum, 'o-', label='Добавили 0.2 I')
axes[0].set(xlabel='Номер направления', ylabel='Собственное значение',
            title=f'15 строк, 24 столбца, ранг {rank}')
axes[0].legend()
# Изменяем веса вдоль слабого направления, точная ошибка известна.
offsets = np.linspace(0, .4, 101)
optimal_J = objective(X, y, w, alpha=lambda2, l1_ratio=0.)
excess = [objective(X, y, w + t * qminus, alpha=lambda2,
                    l1_ratio=0.) - optimal_J for t in offsets]
mu = eigen[0] + lambda2
bounds = np.sqrt(2 * np.maximum(excess, 0.) / mu)
axes[1].plot(offsets, bounds, label='Верхняя граница из цели', lw=3)
axes[1].plot(offsets, offsets, '--', label='Фактическое расстояние')
axes[1].set(xlabel='Сдвиг весов по слабому направлению',
            ylabel='Расстояние до точного Ridge')
axes[1].legend(fontsize=11)
savefig(fig, 'e16_curvature')

D = [[-1., -1.], [1., 1.]]
dy = [-3., 3.]
ends = [objective(D, dy, t, alpha=1., l1_ratio=.2)
        for t in [[2., 0.], [0., 2.], [1., 1.]]]
zero_lasso = coordinate_descent(D, [0., 0.], .2, 1., w0=[1., 2.])
assert zero_lasso['coef'] == [0., 0.]
RESULT = {'eigenvalues': eigen.tolist(),
          'condition_before': float(eigen[-1] / eigen[0]),
          'condition_after': float((eigen[-1] + .09) / (eigen[0] + .09)),
          'ols_shift': float(np.linalg.norm(ols_new - ols)),
          'ridge_shift': float(np.linalg.norm(shifted - w)),
          'independent_agreement': agreement, 'rank_centered': rank,
          'spectral_shift_error': float(np.max(np.abs(
              shift_spectrum - spectrum - .2))),
          'interpolation_errors_original_permuted': interpolation,
          'midpoint_objectives': ends, 'zero_lasso': zero_lasso['coef'],
          'mu': float(mu), 'ridge_coef': w.tolist(),
          'objective_excess': excess, 'coefficient_bounds': bounds.tolist(),
          'bound_max_error': float(np.max(np.abs(bounds - offsets)))}
FIGURES = ['e16_condition', 'e16_curvature']
finish_extra('e16', RESULT)
