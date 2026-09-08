"""Решение Д06.3: собственная потеря и библиотечные квантили."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import QuantileRegressor
from mae_extra import pinball, quantile_interval
from pure_core import mae
from extra_utils import finish_extra

y = [1, 3, 7]
assert quantile_interval(y, .5) == (3, 3)
assert quantile_interval(y, .75) == (7, 7)
assert pinball(y, [3]*3, .5) == 1
assert abs(pinball(y, [7]*3, .75) - 5/6) < 1e-12
for prediction in [[0, 0, 0], [1, -2, 8], [1, 3, 7]]:
    assert abs(2*pinball(y, prediction, .5) - mae(y, prediction)) < 1e-12
assert quantile_interval([0, 10, 20, 30], .6) == (20, 20)
assert pinball([0, 10, 20, 30], [20]*4, .6) < pinball(
    [0, 10, 20, 30], [18]*4, .6)
rng = np.random.default_rng(404)
X = rng.uniform(0, 5, size=(80, 1))
y_train = 1 + X[:, 0] + rng.normal(size=80)
T = rng.uniform(0, 5, size=(150, 1))
y_test = 1 + T[:, 0] + rng.normal(size=150)
outputs, errors = [], {}
for q in [.1, .5, .9]:
    fit = QuantileRegressor(quantile=q, alpha=0,
                            solver="highs").fit(X, y_train)
    prediction = fit.predict(T)
    outputs.append(prediction)
    errors[str(q)] = pinball(y_test, prediction, q)
lower, median, upper = outputs
RESULT = {"manual_minima": [1, 5/6], "test_pinball": errors,
          "test_coverage": float(np.mean((y_test >= lower)
                                        & (y_test <= upper))),
          "crossings": int(np.sum((lower > median) | (median > upper)))}
finish_extra("solution_e06", RESULT)
