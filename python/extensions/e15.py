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

FIGURES = ['e15_bounds', 'e15_convergence']
X, y = [[-1.], [1.]], [-2., 2.]
a = .5
manual = dual_report(X, y, [1.], alpha=a)
weights = [i/50 for i in range(-25, 151)]
reports = [dual_report(X, y, [w], alpha=a) for w in weights]
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.plot(weights, [d['primal'] for d in reports], label='P(w)')
ax.plot(weights, [d['dual'] for d in reports], label='D(theta)')
ax.axhline(.875, color='gray', linestyle='--', label='P* = 0,875')
ax.set(xlabel='Пробный коэффициент w', ylabel='Критерий')
ax.legend()
savefig(fig, FIGURES[0])
fit = ista(X, y, alpha=a, step=.35, max_iter=60, tol=1e-10)
track = [dual_report(X, y, w, alpha=a)
         for w in fit['coef_history']]
kkt = [kkt_report(X, y, w, alpha=a)['max_violation']
       for w in fit['coef_history']]
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.semilogy([max(d['gap'], 1e-15) for d in track],
            label='Двойственный зазор')
ax.semilogy([max(v, 1e-15) for v in kkt], label='Невязка KKT')
ax.set(xlabel='Шаг ISTA', ylabel='Значение; порог рисунка 1e−15')
ax.legend()
savefig(fig, FIGURES[1])
shifted = dual_report([[9.], [11.]], [8., 12.], [1.], 2., a)
no_penalty = dual_report([[1.], [1.]], [-1., 1.],
                         [0.], alpha=0.)
RESULT = dict(manual=manual, track=track, kkt=kkt,
              shifted_wrong_intercept=shifted,
              no_penalty=no_penalty, status=fit['status'])
assert abs(manual['primal']-1.) < 1e-14
assert abs(manual['dual']-.875) < 1e-14
assert abs(manual['gap']-.125) < 1e-14
finish_extra('e15', RESULT)
