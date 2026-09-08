"""Д12.3. Самостоятельный путь и пересчёт единиц ответа."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
from pure_core import coordinate_descent, predict, mse
from extra_utils import finish_extra

X = [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]
y = [2 * row[0] + .5 * row[1] for row in X]
path = []
for alpha in [.01, .05, .2, .8, 2., 10.]:
    model = coordinate_descent(X, y, alpha, .5, tol=1e-11)
    assert model['converged']
    w = model['coef']
    P = .5 * sum(abs(v) for v in w) + .25 * sum(v * v for v in w)
    error = mse(y, predict(X, w, model['intercept']))
    path.append({'alpha': alpha, 'P': P, 'alpha_P': alpha * P,
                 'train_mse': error, 'coef': w})
assert all(b['P'] <= a['P'] + 1e-10 for a, b in zip(path, path[1:]))
assert all(b['train_mse'] >= a['train_mse'] - 1e-10
           for a, b in zip(path, path[1:]))
l1, l2, factor = .3, .2, 10.
base = coordinate_descent(X, y, l1 + l2, l1 / (l1 + l2))
new_alpha = factor * l1 + l2
scaled = coordinate_descent(X, [factor * yi for yi in y],
                            new_alpha, factor * l1 / new_alpha)
error = max(abs(v - factor * w)
            for v, w in zip(scaled['coef'], base['coef']))
assert error < 1e-8
RESULT = {'path': path, 'rotated_L1': 2 * math.sqrt(2),
          'rotated_L2_squared': 4., 'new_alpha': new_alpha,
          'new_rho': factor * l1 / new_alpha, 'scaling_error': error}
finish_extra('solution_e12', RESULT)
