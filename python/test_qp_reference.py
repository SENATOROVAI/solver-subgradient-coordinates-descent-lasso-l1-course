"""Независимый QP-эталон и сертификаты в 14 неоптимальных точках."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from pure_core import elasticnet_cd
from optim_extra import dual_report
rng = np.random.default_rng(926)
X = rng.normal(size=(24, 5))
X[:, 4] = X[:, 0] + .15 * rng.normal(size=24)
X -= X.mean(axis=0)
y = X @ np.array([2., -1., .4, 0., 0.]) + rng.normal(0, .2, 24)
y -= y.mean()
n, p = X.shape
l1, l2 = .13, .07

def objective(z):
    w, t = z[:p], z[p:]
    return np.mean((y - X @ w) ** 2) / 2 + l1 * t.sum() + l2 * w @ w / 2

def gradient(z):
    return np.r_[X.T @ (X @ z[:p] - y) / n + l2 * z[:p],
                 np.full(p, l1)]

# t >= w, t >= -w заменяют абсолютные значения линейными границами.
C = np.block([[-np.eye(p), np.eye(p)], [np.eye(p), np.eye(p)]])
fit = minimize(objective, np.zeros(2 * p), jac=gradient,
               constraints=LinearConstraint(C, 0., np.inf),
               bounds=Bounds(np.r_[np.full(p, -np.inf), np.zeros(p)],
                             np.full(2 * p, np.inf)),
               method='SLSQP', options={'ftol': 1e-12, 'maxiter': 5000})
assert fit.success, fit.message
own = elasticnet_cd(X, y, alpha=l1 + l2, l1_ratio=l1 / (l1 + l2),
                    max_iter=50000, tol=1e-10)
assert own['converged']
error = float(np.max(abs(np.array(own['coef']) - fit.x[:p])))
assert error < 2e-6
certificates = []
for _ in range(14):
    w = rng.normal(size=p)
    check = dual_report(X, y, w, alpha=l1 + l2,
                        l1_ratio=l1 / (l1 + l2))
    assert check['dual'] <= fit.fun + 1e-9
    assert fit.fun <= check['primal'] + 1e-9
    assert check['feasibility_error'] < 1e-12
    certificates.append({'primal': check['primal'], 'dual': check['dual'],
                         'gap': check['gap']})
RESULT = {'status': 'passed', 'qp_success': bool(fit.success),
          'qp_objective': float(fit.fun), 'max_coef_difference': error,
          'independent_points': certificates}
path = Path(__file__).parent / 'extra_results/qp_reference.json'
path.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(RESULT))
