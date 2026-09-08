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

X = [[-1., -2.], [-1., 2.], [1., -2.], [1., 2.]]
y = [2*a+.25*b for a, b in X]
l1, l2 = .2, .8
correct = [1., 1/6]
wrong = [(2-l1)/(1+2*l2), (1-l1)/(4+2*l2)]
wrong_kkt = kkt_report(X, y, wrong, alpha=1., l1_ratio=.2)
assert wrong_kkt['max_violation'] > .5
zero = [[0., 0.]]*4
penalized = ista(zero, [1., 2., 3., 4.], alpha=1.,
                 l1_ratio=.2, w0=[3., -4.])
free = ista(zero, [1., 2., 3., 4.], alpha=0., w0=[3., -4.])
blocked = ista(zero, [1., 2., 3., 4.], alpha=1.,
               l1_ratio=.2, w0=[3., -4.], max_iter=0)
assert penalized['coef'] == [0., 0.]
assert free['coef'] == [3., -4.] and free['n_iter'] == 0
assert blocked['coef'] == [3., -4.] and not blocked['converged']
assert abs((1/4)/(1+.8/4)-1/4.8) < 1e-14
RESULT = dict(correct=correct, double_l2_coef=wrong,
              double_l2_kkt=wrong_kkt, zero_penalized=penalized,
              zero_free=free, zero_budget=blocked)
finish_extra('solution_e20', RESULT)
