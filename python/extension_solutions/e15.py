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

X, y = [[-1.], [1.]], [-2., 2.]
start = dual_report(X, y, [0.], alpha=.5)
base = dual_report(X, y, [1.], alpha=.5)
shift = dual_report(X, [v+10 for v in y], [1.], 10., alpha=.5)
for key in ['primal', 'dual', 'gap']:
    assert abs(base[key]-shift[key]) < 1e-14
assert abs(sum(shift['theta'])) < 1e-14
assert abs(start['gap']-1.125) < 1e-14
RESULT = dict(start=start, base=base, shifted=shift)
finish_extra('solution_e15', RESULT)
