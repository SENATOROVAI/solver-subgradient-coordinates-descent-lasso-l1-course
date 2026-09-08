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

X, y = [[-1., -1.], [1., 1.]], [-3., 3.]
w, l1 = [.5, .5], .2
rows = []
for l2 in [.1, .4, .8, 2.]:
    alpha, rho = l1+l2, l1/(l1+l2)
    optimum = (3-l1)/(2+l2)
    d = dual_report(X, y, w, alpha=alpha, l1_ratio=rho)
    k = kkt_report(X, y, w, alpha=alpha, l1_ratio=rho)
    gnorm = math.sqrt(sum(v*v for v in k['coordinate_violations']))
    exact = math.hypot(w[0]-optimum, w[1]-optimum)
    pstar = objective(X, y, [optimum]*2, alpha=alpha,
                       l1_ratio=rho)
    # Здесь theta=r/n и X^T theta=(2,2).
    gstar = 2*soft_threshold(2., l1)**2/(2*l2)
    fenchel = l1*sum(abs(v) for v in w)
    fenchel += l2*sum(v*v for v in w)/2+gstar
    fenchel -= 2*sum(w)
    assert abs(d['gap']-fenchel) < 1e-13
    assert exact <= d['coefficient_bound']+1e-12
    assert exact <= gnorm/l2+1e-12
    assert d['primal']-pstar <= gnorm*gnorm/(2*l2)+1e-12
    rows.append(dict(lambda2=l2, optimum=optimum,
                     exact_distance=exact, gap=d['gap'],
                     relative_gap=d['gap']/max(1., abs(d['primal'])),
                     gap_bound=d['coefficient_bound'],
                     kkt_bound=gnorm/l2))
RESULT = dict(rows=rows)
finish_extra('solution_e21', RESULT)
