"""Д12. Бюджет, поворот признаков и единицы отклика."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import numpy as np  # Явный мост: сетка и поворот матрицы.
import matplotlib.pyplot as plt
from pure_core import coordinate_descent, objective, predict
from plot_utils import savefig
from extra_utils import finish_extra

X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], float)
y = X @ np.array([2.0, 0.5])
alpha, rho = 0.8, 1.0
model = coordinate_descent(X, y, alpha, rho, fit_intercept=False)
w = model['coef']
budget = sum(abs(v) for v in w)
axis = np.linspace(-0.2, 2.3, 251)
u, v = np.meshgrid(axis, axis)
loss = ((u - 2) ** 2 + (v - 0.5) ** 2) / 2
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.contour(u, v, loss, levels=[0.15, .445, .8, 1.5], colors='#24577C')
diamond = np.array([[budget, 0], [0, budget], [-budget, 0],
                    [0, -budget], [budget, 0]])
ax.fill(diamond[:, 0], diamond[:, 1], color='#469278', alpha=.15)
ax.plot(*diamond.T, label=f'Бюджет {budget:.1f}')
ax.scatter(*w, s=75, color='#CE7334', label='Со штрафом')
ax.scatter(2, .5, marker='x', s=75, label='Без штрафа')
ax.set(xlabel='Первый вес', ylabel='Второй вес', xlim=(-1.4, 2.4),
       ylim=(-1.4, 1.7), title='Один минимум: штраф и найденный бюджет')
ax.set_aspect('equal')
ax.legend(fontsize=11, loc='upper center',
          bbox_to_anchor=(.5, -.17), ncol=3)
savefig(fig, 'e12_budget')

angle = math.pi / 4
Q = np.array([[math.cos(angle), -math.sin(angle)],
              [math.sin(angle), math.cos(angle)]])
rotated = X @ Q
rows = []
for name, mix in [('Lasso', 1.0), ('Ridge', 0.0), ('ElasticNet', .5)]:
    original = coordinate_descent(X, y, .4, mix, tol=1e-12)
    transformed = coordinate_descent(rotated, y, .4, mix, tol=1e-12)
    raw_w = Q @ np.array(transformed['coef'])
    error = float(np.max(np.abs(X @ original['coef'] - X @ raw_w)))
    rows.append({'name': name, 'original': original['coef'],
                 'rotated_back': raw_w.tolist(), 'prediction_gap': error})
assert rows[1]['prediction_gap'] < 1e-10
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.5))
for j, r in enumerate(rows):
    axes[0].scatter(j - .08, r['original'][0], s=65, color='#24577C')
    axes[0].scatter(j + .08, r['rotated_back'][0], s=65, color='#CE7334')
axes[0].scatter([], [], color='#24577C', label='Исходные признаки')
axes[0].scatter([], [], color='#CE7334', label='После поворота и возврата')
axes[0].set(xticks=range(3), xticklabels=[r['name'] for r in rows],
            ylabel='Первый вес в исходных единицах')
axes[0].legend(fontsize=11)
axes[1].bar([r['name'] for r in rows], [r['prediction_gap'] for r in rows])
axes[1].set(ylabel='Максимальное расхождение прогнозов',
            title='Обучение с теми же штрафами')
savefig(fig, 'e12_rotation')

# Меняем единицы y, сохраняя эквивалентность всей цели.
lambda1, lambda2, factor = .2, .3, 100.0
base = coordinate_descent(X, y, lambda1 + lambda2,
                          lambda1 / (lambda1 + lambda2), tol=1e-12)
new_alpha = factor * lambda1 + lambda2
new_rho = factor * lambda1 / new_alpha
changed = coordinate_descent(X, factor * y, new_alpha, new_rho,
                             tol=1e-10)
scaled_error = float(np.max(np.abs(np.array(changed['coef'])
                                   - factor * np.array(base['coef']))))
assert scaled_error < 1e-8
J = objective(X, y, base['coef'], base['intercept'], .5, .4)
Jnew = objective(X, factor * y, changed['coef'], changed['intercept'],
                 new_alpha, new_rho)
assert abs(Jnew / J - factor ** 2) < 1e-7
# При фиксированной форме штрафа сумма P убывает, train MSE растет.
path = []
for strength in np.geomspace(.01, 10., 60):
    candidate = coordinate_descent(X, y, strength, .5, tol=1e-12)
    coef = candidate['coef']
    penalty = .5 * sum(abs(t) for t in coef)
    penalty += .25 * sum(t * t for t in coef)
    error = float(np.mean((y - X @ coef) ** 2))
    path.append([float(strength), penalty, error, strength * penalty])
path = np.array(path)
assert np.all(np.diff(path[:, 1]) <= 1e-10)
assert np.all(np.diff(path[:, 2]) >= -1e-10)
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.5))
axes[0].semilogx(path[:, 0], path[:, 1])
axes[0].set(xlabel='Общая сила α при доле L1 = 0.5',
            ylabel='Форма штрафа P(w), без множителя α')
axes[1].semilogx(path[:, 0], path[:, 2])
axes[1].set(xlabel='Общая сила α при доле L1 = 0.5',
            ylabel='Обучающая MSE')
savefig(fig, 'e12_path_order')
RESULT = {'budget': budget, 'penalty_alpha': alpha, 'coef': w,
          'rotation': rows, 'y_factor': factor, 'new_alpha': new_alpha,
          'new_rho': new_rho, 'coefficient_scaling_error': scaled_error,
          'objective_ratio': Jnew / J, 'path': path.tolist()}
FIGURES = ['e12_budget', 'e12_rotation', 'e12_path_order']
finish_extra('e12', RESULT)
