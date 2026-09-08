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
base = alpha_max(X, y, l1_ratio=.25)
repeated = alpha_max(X*2, y*2, l1_ratio=.25)
scaled = alpha_max(X, [-3*v for v in y], l1_ratio=.25)
shifted = alpha_max(X, [v+100 for v in y], l1_ratio=.25)
assert (base, repeated, scaled, shifted) == (8., 8., 24., 8.)
D = [[-1., -1.], [1., 1.]]
target = [-2., 2.]
cd = elasticnet_cd(D, target, alpha=0., l1_ratio=.25, tol=1e-12)
minnorm = [1., 1.]
pred = [sum(a*b for a, b in zip(row, cd['coef'])) for row in D]
assert max(abs(a-b) for a, b in zip(pred, target)) < 1e-12
ridge_rejected = False
try:
    alpha_max(X, y, l1_ratio=0.)
except ValueError:
    ridge_rejected = True
assert ridge_rejected
RESULT = dict(cutoffs=[base, repeated, scaled, shifted],
              cd_coef=cd['coef'], minimum_norm=minnorm,
              predictions=pred, ridge_rejected=ridge_rejected)
finish_extra('solution_e19', RESULT)
