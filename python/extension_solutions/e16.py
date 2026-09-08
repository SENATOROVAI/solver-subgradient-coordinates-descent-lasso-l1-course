"""Д16.3. Расширенный МНК и интерполяция при числе весов больше строк."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import numpy as np  # Явный численный эталон.
from sklearn.linear_model import Ridge
from extra_utils import finish_extra

rng = np.random.default_rng(813)
x = rng.normal(size=(12, 2))
X = np.column_stack([x, x[:, 0]])
X -= X.mean(axis=0)
y = 2 * X[:, 0] - X[:, 1] + .1 * rng.normal(size=len(X))
y -= y.mean()
lambda2 = .2
aug_X = np.vstack([X, math.sqrt(len(y) * lambda2) * np.eye(3)])
aug_y = np.r_[y, np.zeros(3)]
w = np.linalg.lstsq(aug_X, aug_y, rcond=None)[0]
reference = Ridge(alpha=len(y) * lambda2,
                  fit_intercept=False).fit(X, y)
error = float(np.max(np.abs(w - reference.coef_)))
assert error < 1e-10
wide = rng.normal(size=(8, 12))
wide -= wide.mean(axis=0)
random_y = rng.permutation(rng.normal(size=8))
random_y -= random_y.mean()
solution = np.linalg.lstsq(wide, random_y, rcond=None)[0]
interpolation_error = float(np.max(np.abs(wide @ solution - random_y)))
assert interpolation_error < 1e-10
RESULT = {'manual_mu': .2, 'manual_condition': 2.2 / .2,
          'manual_distance_bound': math.sqrt(2 * .001 / .2),
          'augmented_ridge_error': error,
          'wide_rank': int(np.linalg.matrix_rank(wide)),
          'interpolation_error': interpolation_error}
finish_extra('solution_e16', RESULT)
