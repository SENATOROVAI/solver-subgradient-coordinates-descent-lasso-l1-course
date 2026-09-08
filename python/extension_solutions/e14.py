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

x, z = [-1., -1., 1., 1.], [-1., 1., -1., 1.]
X = [[a, .8*a+.6*b] for a, b in zip(x, z)]
y = [7*a/9+5*b/18 for a, b in X]
rows = []
for decay in [0., .6]:
    fit = lasso_subgradient(X, y, alpha=.2, step=.5,
                            decay=decay, max_iter=300)
    rows.append(dict(decay=decay, last=fit['objective'],
                     best=fit['best_objective'],
                     kkt=fit['optimality_error'], status=fit['status'],
                     increases=sum(b > a+1e-14 for a, b in
                                  zip(fit['history'], fit['history'][1:]))))
    assert fit['best_objective'] <= fit['objective']+1e-14
pairs = [(a, b) for a in [-.51, -.5, 0., .46, .55, 1.]
         for b in [-.51, -.5, 0., .46, .55, 1.]]
assert all(abs(soft_threshold(a, .5)-soft_threshold(b, .5))
           <= abs(a-b)+1e-14 for a, b in pairs)
soft = [soft_threshold(v, .5) for v in [.46, .55]]
hard = [v if abs(v) > .5 else 0. for v in [.46, .55]]
# В положительном секторе постоянный шаг здесь удачен.
# Отдельный пример Д14 показывает отсутствие общей монотонности.
RESULT = dict(rows=rows, soft=soft, hard=hard,
              soft_distance=abs(soft[1]-soft[0]),
              hard_distance=abs(hard[1]-hard[0]))
finish_extra('solution_e14', RESULT)
