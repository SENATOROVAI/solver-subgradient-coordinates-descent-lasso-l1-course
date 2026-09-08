"""Решение Д04.3: один шаг и несмещённые взвешенные оценки."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mae_extra import stochastic_mae, mae_gradient
from pure_core import predict, mae
from extra_utils import finish_extra

X, y = [[0], [1], [2]], [1, 2, 2]
gw, gb = mae_gradient(X, y, [0], 0)
w, b = [-.2*gw[0]], -.2*gb
assert w == [.2] and b == .2
assert abs(mae(y, predict(X, w, b)) - 19/15) < 1e-12
weighted_gw, weighted_gb = mae_gradient(
    [[1], [2]], [1, -1], [0], 0, weights=[1, 3])
assert weighted_gw == [1.25] and weighted_gb == .5
uniform_expectation = (-.5 + 3) / 2
weighted_expectation = .25*(-1) + .75*2
assert uniform_expectation == weighted_expectation == 1.25
fit = stochastic_mae(X, y, updates=1000, batch_size=2,
                     step=.2, power=.6, seed=103, evaluate_every=10)
assert fit["best_loss"] <= fit["last_loss"] + 1e-12
assert fit["update_rows"] == 2000
assert fit["diagnostic_rows"] == 303
assert not fit["converged"]
RESULT = {"new_coef": w, "new_intercept": b, "step_mae": 19/15,
          "weighted_gradient": weighted_gw,
          "weighted_intercept_gradient": weighted_gb,
          "last_loss": fit["last_loss"], "best_loss": fit["best_loss"],
          "average_loss": fit["average_loss"],
          "update_rows": fit["update_rows"],
          "diagnostic_rows": fit["diagnostic_rows"],
          "status": fit["status"]}
finish_extra("solution_e04", RESULT)
