"""Решение Д25.3: известная Lasso-задача на трёх уровнях."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import Lasso, ElasticNet, _cd_fast, enet_path
from extra_utils import finish_extra

X = np.asfortranarray([[1., 1.], [-1., 1.],
                       [1., -1.], [-1., -1.]])
y = np.array([3., -1., 1., -3.])
settings = dict(alpha=.5, fit_intercept=False, tol=1e-12,
                 max_iter=30000)
lasso = Lasso(**settings).fit(X, y)
elastic = ElasticNet(l1_ratio=1., **settings).fit(X, y)
raw = _cd_fast.enet_coordinate_descent(
    np.zeros(2), 4 * .5, 0., X, y, 30000, 1e-12,
    np.random.RandomState(1621))
assert Lasso.fit is ElasticNet.fit and Lasso.path is enet_path
for coefficient in [lasso.coef_, elastic.coef_, raw[0]]:
    assert np.allclose(coefficient, [1.5, .5], atol=1e-10)
assert abs(lasso.dual_gap_ - raw[1] / len(y)) < 1e-10
r = y - X @ raw[0]
objective = r @ r / 8 + .5 * np.abs(raw[0]).sum()
assert np.isclose(objective, 1.25)
RESULT = finish_extra("solution_e25", {
    "coef": raw[0].tolist(), "objective": float(objective),
    "public_gap": float(lasso.dual_gap_),
    "internal_gap": float(raw[1]),
    "manual_internal_penalties": [.5, 1.5],
    "manual_update": -11 / 23, "manual_public_gap": .04,
})
