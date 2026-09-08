"""Д07.3. Единицы, CDF и вырожденный масштаб."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extra_utils import finish_extra
from stat_extra import (laplace_cdf, laplace_pdf,
                        location_scale_mle, trapezoid)

y = [1, 2, 6]
fit = location_scale_mle(y)
seconds = location_scale_mle([60 * v for v in y])
for key in ["normal_mu", "normal_sigma", "laplace_mu", "laplace_scale"]:
    assert abs(seconds[key] - 60 * fit[key]) < 1e-10
for key in ["normal_nll", "laplace_nll"]:
    assert abs(seconds[key] - fit[key] - 3 * math.log(60)) < 1e-10
prob_min = laplace_cdf(3, 2, 1) - laplace_cdf(1, 2, 1)
prob_sec = laplace_cdf(180, 120, 60) - laplace_cdf(60, 120, 60)
assert abs(prob_min - prob_sec) < 1e-12
grid = [-2 + i / 10000 for i in range(40001)]
area = trapezoid(grid, [laplace_pdf(v, 0, 0.1) for v in grid])
assert abs(area - 1) < 1e-6
try:
    location_scale_mle([2, 2, 2])
    raise AssertionError("Вырожденная выборка должна отклоняться")
except ValueError:
    degenerate = "Нет положительной конечной MLE масштаба"
RESULT = {"manual": fit, "probability_both_units": prob_min,
          "area_scale_point1": area, "degenerate": degenerate,
          "nll_unit_change": 3 * math.log(60)}
finish_extra("solution_e07", RESULT)
