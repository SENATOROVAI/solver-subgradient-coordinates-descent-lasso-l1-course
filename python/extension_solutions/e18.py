"""Д18.3. Границы prior, нормировка и смена единиц; SciPy."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy.integrate import quad
from extra_utils import finish_extra
from extension_listings.en_prior_bridge import log_z, prior_pdf

assert abs(math.exp(log_z(1, 0)) - 2) < 1e-12
assert abs(math.exp(log_z(0, 1)) - math.sqrt(2 * math.pi)) < 1e-12
try:
    log_z(0, 0)
    raise AssertionError("Пара (0,0) должна отклоняться")
except ValueError:
    rejected = True
reports = []
for a, d in [(2, 0.5), (1, 1), (100, 1)]:
    area = quad(lambda t: 2 * prior_pdf(t, a, d), 0, math.inf)[0]
    assert abs(area - 1) < 1e-8
    reports.append({"a": a, "d": d, "Z": math.exp(log_z(a, d)),
                    "area": area})
# u = c*t: параметры a/c,d/c² и якобиан 1/c.
c, a, d = 3.0, 2.0, 0.5
errors = [abs(prior_pdf(u, a / c, d / c ** 2)
              - prior_pdf(u / c, a, d) / c)
          for u in [-3, -1, 0, 1, 3]]
assert max(errors) < 1e-12
n, sigma2, alpha, rho = 50, 2, 0.3, 0.4
l1, l2 = alpha * rho, alpha * (1 - rho)
RESULT = {"integrals": reports, "zero_pair_rejected": rejected,
          "unit_change_error": max(errors),
          "manual": {"lambda1": l1, "lambda2": l2,
                     "a": n * l1 / sigma2, "d": n * l2 / sigma2,
                     "alpha_double_n_fixed_prior": alpha / 2}}
finish_extra("solution_e18", RESULT)
