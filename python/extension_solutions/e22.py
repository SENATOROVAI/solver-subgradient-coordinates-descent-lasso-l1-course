"""Решение Д22.3: ручной ответ, копии и нулевой вес."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weighted_extra import weighted_cd, prepare_weighted
from pure_core import coordinate_descent
from extra_utils import finish_extra

X, y, s = [[0.], [2.]], [1., 5.], [1., 3.]
settings = dict(alpha=.6, l1_ratio=1 / 3, tol=1e-12)
a = weighted_cd(X, y, s, **settings)
b = coordinate_descent([[0.], [2.], [2.], [2.]],
                        [1., 5., 5., 5.], **settings)
c = weighted_cd(X + [[1000.]], y + [-999.], s + [0.],
                 **settings)
for fit in [a, b, c]:
    assert fit["converged"]
    assert abs(fit["coef"][0] - 26 / 23) < 1e-10
    assert abs(fit["intercept"] - 53 / 23) < 1e-10
for invalid in [[0., 0.], [-1., 2.]]:
    try:
        prepare_weighted(X, y, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Некорректные веса не отклонены")
RESULT = finish_extra("solution_e22", {
    "coef": a["coef"], "intercept": a["intercept"],
    "normalized_weights": prepare_weighted(X, y, s)["q"],
    "weighted_residual_sum": sum(
        v * (t - row[0] * a["coef"][0] - a["intercept"])
        for row, t, v in zip(X, y, s)),
})
