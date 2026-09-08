"""Д17. Граница группировки, число копий и предел малого L2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import numpy as np  # Явный мост для генератора и графических массивов.
import matplotlib.pyplot as plt
from pure_core import coordinate_descent, scale_fit, scale_transform
from geometry_extra import grouping_pair
from plot_utils import savefig
from extra_utils import finish_extra

rng = np.random.default_rng(714)
z = rng.normal(size=90)
raw = np.column_stack([z + .12 * rng.normal(size=90),
                       z + .12 * rng.normal(size=90)])
y = 3 * z + .3 * rng.normal(size=90)
y -= y.mean()
X = scale_transform(raw, scale_fit(raw))
lambda1 = .2
values = np.geomspace(.03, 1., 18)
records = []
for lambda2 in values:
    fit = coordinate_descent(X, y, lambda1 + lambda2,
                             lambda1 / (lambda1 + lambda2),
                             tol=1e-10, max_iter=30000)
    assert fit['converged']
    check = grouping_pair(X, y, fit['coef'], 0, 1, lambda2)
    assert check['valid']
    assert check['difference'] <= check['direct_bound'] + 1e-7
    assert abs(check['direct_bound'] - check['correlation_bound']) < 1e-9
    records.append({'lambda2': float(lambda2), **check})
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.5))
axes[0].loglog(values, [r['difference'] for r in records],
               label='Фактическая разность')
axes[0].loglog(values, [r['direct_bound'] for r in records],
               label='Верхняя граница')
axes[0].set(xlabel='Квадратичная сила λ₂', ylabel='Разность весов')
axes[0].legend(fontsize=11)
axes[1].bar(['Разность', 'Граница'],
            [records[-1]['difference'], records[-1]['direct_bound']])
axes[1].set(ylabel='Величина при λ₂ = 1', title='Оценка имеет запас')
savefig(fig, 'e17_pair_bound')

copies = [1, 2, 4, 8]
copy_rows = []
for count in copies:
    duplicate_X = [[-1.] * count, [1.] * count]
    fit = coordinate_descent(duplicate_X, [-3., 3.], 1., .2,
                             tol=1e-11, max_iter=30000)
    expected = 2.8 / (count + .8)
    assert fit['converged']
    assert max(abs(w - expected) for w in fit['coef']) < 1e-8
    copy_rows.append({'copies': count, 'each': expected,
                      'sum': sum(fit['coef'])})
# Для исчезающего L2 используем точную формулу, а не недосошедший CD.
small = np.geomspace(.0001, .8, 100)
limit_weights = 2.8 / (2 + small)
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.5))
axes[0].plot(copies, [r['sum'] for r in copy_rows], 'o-')
axes[0].axhline(2.8, ls='--', label='Сумма чистого Lasso')
axes[0].set(xticks=copies, xlabel='Число одинаковых столбцов',
            ylabel='Суммарный вес', title='Те же силы: L1 = 0.2, L2 = 0.8')
axes[0].legend(fontsize=11)
axes[1].semilogx(small, limit_weights, label='Каждый вес ElasticNet')
axes[1].axhline(1.4, ls='--', label='Предел: середина Lasso-отрезка')
axes[1].set(xlabel='Положительная сила λ₂', ylabel='Вес каждой копии')
axes[1].legend(fontsize=11)
savefig(fig, 'e17_copies_limit')

# Масштаб и знак: меняется область применения корреляционной формулы.
scaled = [[2 * row[0], row[1]] for row in X]
scaled_fit = coordinate_descent(scaled, y, .5, .4, tol=1e-10)
scaled_check = grouping_pair(scaled, y, scaled_fit['coef'], 0, 1, .3)
assert scaled_check['valid'] and not scaled_check['normalized']
assert scaled_check['correlation_bound'] is None
negative = [[row[0], -row[1]] for row in X]
negfit = coordinate_descent(negative, y, .5, .4, tol=1e-10)
unoriented = grouping_pair(negative, y, negfit['coef'], 0, 1, .3)
assert not unoriented['valid']
oriented_w = [negfit['coef'][0], -negfit['coef'][1]]
oriented = grouping_pair(X, y, oriented_w, 0, 1, .3)
assert oriented['valid']
assert oriented['difference'] <= oriented['direct_bound'] + 1e-7
manual_axis = math.sqrt(5) - 1
manual_diagonal = math.sqrt(3) - 1
for a, b in [(manual_axis, 0.), (manual_diagonal, manual_diagonal)]:
    level = .5 * (abs(a) + abs(b)) + .25 * (a * a + b * b)
    assert abs(level - 1.) < 1e-12
RESULT = {'correlation': float(np.array(X)[:, 0] @ np.array(X)[:, 1] / 90),
          'pair_checks': records, 'copies': copy_rows,
          'manual_contour_axis': manual_axis,
          'manual_contour_diagonal': manual_diagonal,
          'limit_at_smallest': float(limit_weights[0]),
          'scaled_pair': scaled_check, 'negative_raw_pair': unoriented,
          'negative_oriented_pair': oriented,
          'manual_bound': .6 / .3 * math.sqrt(2 * (1 - .98))}
FIGURES = ['e17_pair_bound', 'e17_copies_limit']
finish_extra('e17', RESULT)
