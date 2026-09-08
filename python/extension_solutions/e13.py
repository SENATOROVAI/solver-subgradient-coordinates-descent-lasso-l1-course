from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
from pure_core import (
    objective, soft_threshold, lasso_cd, elasticnet_cd,
)
from optim_extra import (
    kkt_report, alpha_max, ista, lasso_subgradient, dual_report,
)
from extra_utils import finish_extra

X = [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]
y = [5+2*a+.4*b for a, b in X]
base = lasso_cd(X, y, alpha=.5, tol=1e-12)
shift = lasso_cd(X, [v+100 for v in y], alpha=.5, tol=1e-12)
constant = lasso_cd(X, [105.]*4, alpha=0., tol=1e-12)
multipliers = [((.5+c)/2, (.5-c)/2) for c in [.5, -.5, .4]]
assert max(abs(a-b) for a, b in zip(base['coef'], shift['coef'])) < 1e-12
assert abs(shift['intercept']-base['intercept']-100) < 1e-12
assert alpha_max(X, y) == alpha_max(X, [v+100 for v in y])
assert alpha_max(X, [105.]*4) == 0.
assert constant['coef'] == [0., 0.]
RESULT = dict(shift_coef=shift['coef'],
              shift_intercept=shift['intercept'],
              constant_coef=constant['coef'],
              constant_intercept=constant['intercept'],
              multipliers=multipliers)
finish_extra('solution_e13', RESULT)
