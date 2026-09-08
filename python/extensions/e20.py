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

FIGURES = ['e20_paths', 'e20_majorant']
X = [[-1., -2.], [-1., 2.], [1., -2.], [1., 2.]]
y = [2*a+.25*b for a, b in X]
args = dict(alpha=1., l1_ratio=.2, tol=1e-10, max_iter=150)
A = ista(X, y, split='smooth_l2', **args)
B = ista(X, y, split='prox_all', **args)
equal = ista(X, y, split='prox_all', step=A['step'], **args)
path_error = max(abs(u-v) for wa, wb in
                 zip(A['coef_history'], B['coef_history'])
                 for u, v in zip(wa, wb))
fig, axes = plt.subplots(1, 2, figsize=(9., 4.3))
for label, fit, style in [('A: естественный', A, '-'),
                           ('B: естественный', B, '--'),
                           ('B: тот же численный шаг', equal, ':')]:
    for j, ax in enumerate(axes):
        ax.plot([w[j] for w in fit['coef_history'][:16]],
                style, label=label)
        ax.set(xlabel='Шаг ISTA', ylabel=f'w{j+1}')
axes[0].legend(fontsize=11)
savefig(fig, FIGURES[0])
# Дубликаты: вдоль w=(s,s) гладкая часть равна (3-2s)^2/2.
s = [i/100 for i in range(-20, 351)]
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.plot(s, [(3-2*v)**2/2 for v in s], label='Гладкая часть f')
for t, label in [(.5, 'Q: безопасный шаг 0,5'),
                  (1., 'Q: слишком большой шаг 1')]:
    ax.plot(s, [4.5-6*v+v*v/t for v in s], '--', label=label)
ax.set(xlabel='Общий вес s двух дубликатов', ylabel='f или Q')
ax.legend()
savefig(fig, FIGURES[1])
zero = [[7., -2.]]*4
zero_penalty = ista(zero, [1., 2., 3., 4.], w0=[3., -4.],
                    alpha=1., l1_ratio=.2)
zero_free = ista(zero, [1., 2., 3., 4.], w0=[3., -4.], alpha=0.)
manual = ista(X, y, alpha=1., l1_ratio=.2, step=1/4.8,
              max_iter=2, tol=0.)
RESULT = dict(step_A=A['step'], step_B=B['step'], L0=A['L0'],
              first_A=A['coef_history'][1],
              first_equal_B=equal['coef_history'][1],
              max_path_difference=path_error,
              matched_lengths=[len(A['history']), len(B['history'])],
              spectral_manual_path=manual['coef_history'],
              zero_penalty=zero_penalty, zero_free=zero_free,
              status_A=A['status'], status_B=B['status'])
assert len(A['history']) == len(B['history'])
assert path_error < 1e-14
assert zero_penalty['coef'] == [0., 0.]
assert zero_free['coef'] == [3., -4.] and zero_free['n_iter'] == 0
finish_extra('e20', RESULT)
