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

FIGURES = ['e14_methods', 'e14_paths', 'e14_thresholds']
x = [-1., -1., 1., 1.]
z = [-1., 1., -1., 1.]
X = [[a, .8*a + .6*b] for a, b in zip(x, z)]
y = [7*a/9 + 5*b/18 for a, b in X]
a = .2
cd = lasso_cd(X, y, alpha=a, max_iter=300, tol=1e-10)
pm = ista(X, y, alpha=a, step=1/1.8, max_iter=300,
          tol=1e-10)
sg = lasso_subgradient(X, y, alpha=a, max_iter=300,
                       step=.5, decay=.6)
opt = objective(X, y, [2/3, 1/6], alpha=a)
methods = [('CD: проход', cd), ('ISTA: шаг', pm),
           ('Субградиент: шаг', sg)]
fig, axes = plt.subplots(1, 2, figsize=(9., 4.5))
raw = {}
for label, fit in methods:
    excess = [v-opt for v in fit['history']]
    errors = [kkt_report(X, y, w, alpha=a)['max_violation']
              for w in fit['coef_history']]
    raw[label] = dict(excess=excess, kkt=errors)
    axes[0].semilogy([max(v, 1e-12) for v in excess], label=label)
    axes[1].semilogy([max(v, 1e-12) for v in errors], label=label)
axes[0].set(ylabel='J − J*; порог рисунка 1e−12')
axes[1].set(ylabel='Невязка KKT; порог рисунка 1e−12')
for ax in axes:
    ax.set(xlabel='Номер обновления по легенде')
axes[0].legend(fontsize=11)
savefig(fig, FIGURES[0])
fig, ax = plt.subplots(figsize=(7.4, 4.2))
for label, fit in methods:
    pts = fit['coef_history'][:26]
    ax.plot([w[0] for w in pts], [w[1] for w in pts],
            '.-', label=label)
ax.scatter([2/3], [1/6], marker='*', s=140, c='black',
           label='Точный минимум')
ax.set(xlabel='w1', ylabel='w2')
ax.legend()
savefig(fig, FIGURES[1])
vals = [i/100 for i in range(-200, 201)]
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.plot(vals, [soft_threshold(v, .5) for v in vals],
        label='Мягкий порог')
ax.plot(vals, [min(.5, max(-.5, v)) for v in vals],
        label='Обрезание к интервалу')
ax.plot(vals, [v if abs(v) > .5 else 0 for v in vals],
        linestyle='--', label='Жёсткий порог')
ax.set(xlabel='Вход z', ylabel='Выход')
ax.legend()
savefig(fig, FIGURES[2])
trap = [0.]
for k in range(3):
    w = trap[-1]
    trap.append(w + .5*(.2-w-.5*((w > 0)-(w < 0))))
G = [(u-v)/pm['step'] for u, v in
     zip(pm['coef_history'][-2], pm['coef_history'][-1])]
RESULT = dict(first_ista=pm['coef_history'][1],
              first_cd=cd['coef_history'][1], optimum=[2/3, 1/6],
              optimum_objective=opt, trap=trap,
              trap_J=[(w-.2)**2/2+.5*abs(w) for w in trap],
              sg_increases=sum(b > c+1e-14 for c, b in
                               zip(sg['history'], sg['history'][1:])),
              prox_gradient_norm=math.sqrt(sum(v*v for v in G)),
              summaries={k: {q: v[q] for q in [
                  'coef', 'n_iter', 'status', 'optimality_error']}
                  for k, v in methods}, raw=raw)
assert max(abs(u-v) for u, v in
           zip(pm['coef_history'][1], [4/9, 7/18])) < 1e-14
finish_extra('e14', RESULT)
