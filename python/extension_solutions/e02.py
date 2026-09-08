"""Решение Д02.3: веса, интервал и эквивалентные повторения."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mae_extra import weighted_mae, quantile_interval
from extra_utils import finish_extra

y, a = [1, 2, 8, 9], [1, 1, 1, 5]
interval = quantile_interval(y)
weighted = quantile_interval(y, weights=a)
assert interval == (2, 8)
assert weighted == (9, 9)
assert weighted_mae(y, [5]*4) == 3.5
assert weighted_mae(y, [9]*4, a) == 2
copies = [v for v, count in zip(y, a) for _ in range(count)]
for b in [1, 2, 5, 8, 9, 12]:
    expected = weighted_mae(copies, [b]*len(copies))
    assert abs(weighted_mae(y, [b]*4, a) - expected) < 1e-12
    assert abs(weighted_mae(y, [b]*4, [7*v for v in a])
               - expected) < 1e-12
assert quantile_interval([0, 2, 99, 10], weights=[1, 1, 0, 2]) == (2, 10)
rejected = 0
for invalid in [[0]*4, [-1, 1, 1, 1], [1, 2]]:
    try:
        quantile_interval(y, weights=invalid)
    except ValueError:
        rejected += 1
assert rejected == 3
RESULT = {"ordinary_interval": interval, "weighted_interval": weighted,
          "ordinary_minimum": 3.5, "weighted_minimum": 2,
          "invalid_weights_rejected": rejected}
finish_extra("solution_e02", RESULT)
