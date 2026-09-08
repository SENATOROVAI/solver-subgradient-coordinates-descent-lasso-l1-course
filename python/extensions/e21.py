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

FIGURES = ['e21_bounds', 'e21_conjugate']
X, y = [[-1., -1.], [1., 1.]], [-3., 3.]
a, rho, l1, l2 = 1., .2, .2, .8
fit = elasticnet_cd(X, y, alpha=a, l1_ratio=rho,
                    max_iter=50, tol=1e-12)
rows = []
for w in fit['coef_history']:
    d = dual_report(X, y, w, alpha=a, l1_ratio=rho)
    k = kkt_report(X, y, w, alpha=a, l1_ratio=rho)
    gn = math.sqrt(sum(v*v for v in k['coordinate_violations']))
    r = [t-sum(a*b for a, b in zip(row, w))
         for row, t in zip(X, y)]
    theta = d['theta']
    corr = [sum(row[j]*u for row, u in zip(X, theta))
            for j in range(2)]
    mismatch = sum((v-2*u)**2 for v, u in zip(r, theta))/4
    conjugate = sum(soft_threshold(c, l1)**2 for c in corr)/(2*l2)
    fenchel = l1*sum(abs(v) for v in w)
    fenchel += l2*sum(v*v for v in w)/2+conjugate
    fenchel -= sum(v*c for v, c in zip(w, corr))
    rows.append(dict(**d, kkt=k['max_violation'],
                     true_distance=math.hypot(w[0]-1, w[1]-1),
                     kkt_distance_bound=gn/l2,
                     kkt_objective_bound=gn*gn/(2*l2),
                     mismatch=mismatch, fenchel=fenchel))
fig, axes = plt.subplots(1, 2, figsize=(9., 4.5))
axes[0].plot([r['primal'] for r in rows], label='P(w)')
axes[0].plot([r['dual'] for r in rows], label='D(theta)')
axes[0].axhline(1.7, color='gray', linestyle='--', label='P* = 1,7')
axes[0].set(xlabel='Проход CD', ylabel='Критерий')
for key, label in [('true_distance', 'Фактическая ошибка'),
                    ('coefficient_bound', 'Граница из зазора'),
                    ('kkt_distance_bound', 'Граница из KKT')]:
    # None означает: сырое округление не даёт вещественной границы.
    values = [r[key] for r in rows[:16]]
    axes[1].semilogy([None if v is None else max(v, 1e-14)
                      for v in values], label=label)
axes[1].set(xlabel='Первые 15 проходов CD',
            ylabel='Норма ошибки; порог 1e−14')
for ax in axes:
    ax.legend(fontsize=11)
savefig(fig, FIGURES[0])
vgrid = [i/100 for i in range(-200, 201)]
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.plot(vgrid, [soft_threshold(v, l1)**2/(2*l2)
               for v in vgrid], label='Сопряжённый штраф g*(v)')
ax.axvspan(-l1, l1, color='#469278', alpha=.15,
           label='Нулевой участок')
ax.set(xlabel='Аргумент v', ylabel='Максимум v·w − g(w)')
ax.legend()
savefig(fig, FIGURES[1])
manual = dual_report(X, y, [.5, .5], alpha=a, l1_ratio=rho)
start = dual_report(X, y, [0., 0.], alpha=a, l1_ratio=rho)
RESULT = dict(manual=manual, start=start, rows=rows,
              optimum=[1., 1.], optimum_objective=1.7,
              status=fit['status'],
              max_decomposition_error=max(abs(r['gap']-r['mismatch']
                                              -r['fenchel']) for r in rows))
assert abs(manual['gap']-2.45) < 1e-14
assert abs(start['dual']+5.3) < 1e-14
assert all(r['true_distance'] <= r['kkt_distance_bound']+1e-12
           for r in rows)
assert RESULT['max_decomposition_error'] < 1e-13
finish_extra('e21', RESULT)
