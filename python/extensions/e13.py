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

FIGURES = ['e13_balance', 'e13_threshold']
X = [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]
y = [5 + 2*a + .4*b for a, b in X]
a = .5
fit = lasso_cd(X, y, alpha=a, tol=1e-12)
good = kkt_report(X, y, fit['coef'], fit['intercept'], a)
bad = kkt_report(X, y, [1.4, .02], 5., a)
cutoff = alpha_max(X, y)
grid = [i / 20 for i in range(51)]
path = [lasso_cd(X, y, alpha=v, tol=1e-12) for v in grid]
fig, ax = plt.subplots(figsize=(7.4, 4.1))
ax.scatter([1, 2], good['correlations'], s=90, label='Минимум')
ax.scatter([1, 2], bad['correlations'], marker='x', s=100,
           label='Изменённые веса')
for level in [-a, a]:
    ax.axhline(level, color='#CE7334', linestyle='--')
ax.set(xticks=[1, 2], xlabel='Координата j',
       ylabel='Среднее произведение Xj и остатка')
ax.legend()
savefig(fig, FIGURES[0])
fig, ax = plt.subplots(figsize=(7.4, 4.1))
for j in range(2):
    ax.plot(grid, [v['coef'][j] for v in path], label=f'w{j+1}')
ax.axvline(cutoff, color='gray', linestyle='--',
           label='Порог нулевой модели')
ax.set(xlabel='alpha', ylabel='Вес')
ax.legend()
savefig(fig, FIGURES[1])
boundary = [lasso_cd(X, y, alpha=fac*cutoff, tol=1e-12)
            for fac in [.99, 1., 1.01]]
small = kkt_report([[-1.], [1.]], [-.2, .2], [-1e-10],
                   alpha=.5)
rounded = kkt_report([[-1.], [1.]], [-.2, .2], [0.],
                     alpha=.5)
RESULT = dict(coef=fit['coef'], intercept=fit['intercept'],
              good=good, bad=bad, alpha_max=cutoff,
              boundary=[f['coef'] for f in boundary],
              small_negative=small, actually_zero=rounded,
              zero_cutoff=alpha_max(X, [5.]*4))
assert max(good['coordinate_violations']) < 1e-12
assert cutoff == 2 and boundary[1]['coef'] == [0., 0.]
finish_extra('e13', RESULT)
