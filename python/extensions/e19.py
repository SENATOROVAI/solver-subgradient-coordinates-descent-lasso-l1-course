from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import random
import matplotlib.pyplot as plt
from plot_utils import savefig
from extra_utils import finish_extra
from pure_core import (
    objective, soft_threshold, lasso_cd, elasticnet_cd,
    marketing_data, scale_fit, scale_transform,
)
from optim_extra import (
    kkt_report, alpha_max, ista, lasso_subgradient, dual_report,
)
random.seed(20260907)

FIGURES = ['e19_balance', 'e19_endpoints']
X = [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]
y = [5 + 2*a + .1*b for a, b in X]
a, rho = 1., .3
fit = elasticnet_cd(X, y, alpha=a, l1_ratio=rho, tol=1e-12)
good = kkt_report(X, y, fit['coef'], 5., a, rho)
bad = kkt_report(X, y, [.95, 0.], 5., a, rho)
fig, ax = plt.subplots(figsize=(7.4, 4.2))
for offset, report, w, label in [
        (-.12, good, [1., 0.], 'Минимум'),
        (.12, bad, [.95, 0.], 'Вес w1 = 0,95')]:
    values = [c-.7*v for c, v in zip(report['correlations'], w)]
    ax.bar([1+offset, 2+offset], values, width=.23, label=label)
for level in [-.3, .3]:
    ax.axhline(level, color='gray', linestyle='--')
ax.set(xticks=[1, 2], xlabel='Координата j',
       ylabel='c_j − lambda2 · w_j')
ax.legend()
savefig(fig, FIGURES[0])
grid = [i/10 for i in range(121)]
fig, ax = plt.subplots(figsize=(7.4, 4.2))
for r in [0., .25, .5, 1.]:
    weights = [max(2-v*r, 0)/(1+v*(1-r)) for v in grid]
    ax.plot(grid, weights, label=f'rho = {r:g}')
ax.set(xlabel='alpha', ylabel='Вес первого признака')
ax.legend()
savefig(fig, FIGURES[1])
y2 = [5 + 2*a + .4*b for a, b in X]
cutoff = alpha_max(X, y2, l1_ratio=.25)
boundary = [elasticnet_cd(X, y2, alpha=t*cutoff,
                         l1_ratio=.25, tol=1e-12)['coef']
            for t in [.99, 1., 1.01]]
RESULT = dict(coef=fit['coef'], good=good, bad=bad,
              alpha_max=cutoff, boundary=boundary,
              ridge_at_12=2/13,
              zero_alpha_max=alpha_max(X, [5.]*4, l1_ratio=.25))
assert max(abs(u-v) for u, v in zip(fit['coef'], [1., 0.])) < 1e-14
assert abs(bad['max_violation']-.085) < 1e-14
finish_extra('e19', RESULT)
