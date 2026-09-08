"""Решение Д08.3: одинаковая цель при общем масштабе весов."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn._loss.loss import AbsoluteError
from extra_utils import finish_extra

y = np.array([1., 2., 9.])
s = np.array([1., 1., 5.])
prediction = np.full(3, 2.)
loss = AbsoluteError()
a = loss.loss(y, prediction, sample_weight=s)
b = loss.loss(y, prediction, sample_weight=10 * s)
assert np.allclose(b, 10 * a)
assert np.isclose(loss(y, prediction, sample_weight=s), 36 / 7)
assert np.isclose(loss(y, prediction, sample_weight=10 * s), 36 / 7)
assert loss.fit_intercept_only(y, s) == 9
assert loss.fit_intercept_only(y, 10 * s) == 9
RESULT = finish_extra("solution_e08", {
    "loss_at_2": float(loss(y, prediction, sample_weight=s)),
    "loss_at_9": float(loss(y, np.full(3, 9.), sample_weight=s)),
    "pointwise": a.tolist(), "pointwise_scaled": b.tolist(),
    "median": 9.,
})
