"""Точная дисперсия линейной оценки; NumPy здесь не требуется."""

import json
import math


n, sigma2, rho, penalty = 100, 1., .99, .09
patterns = [(-1., -1.), (-1., 1.), (1., -1.), (1., 1.)]
X = [[a, rho * a + math.sqrt(1 - rho * rho) * b]
     for a, b in patterns * 25]
unit = 1 / math.sqrt(2)
directions = [[unit, -unit], [unit, unit]]
prediction_mse = [sum(sum(a * b for a, b in zip(row, v)) ** 2
                      for row in X) / n for v in directions]
assert abs(prediction_mse[0] - .01) < 1e-14
assert abs(prediction_mse[1] - 1.99) < 1e-14
weak_column = [sum(a * b for a, b in zip(row, directions[0]))
               for row in X]
d = sum(v * v for v in weak_column) / n
# Оценка направления есть sum(h_i*y_i); Var = sigma2*sum(h_i²).
h = [v / (n * (d + penalty)) for v in weak_column]
variance = sigma2 * sum(v * v for v in h)
assert abs(variance - .01) < 1e-14
rows = []
for beta in [0., 2.]:
    expected = sum(a * beta * x for a, x in zip(h, weak_column))
    bias2 = (expected - beta) ** 2
    rows.append(dict(beta=beta, mean=expected, bias2=bias2,
                     variance=variance, mse=bias2 + variance))
assert abs(rows[0]["mse"] - .01) < 1e-14
assert abs(rows[1]["mse"] - 3.25) < 1e-13
RESULT = dict(n=n, sigma2=sigma2, lambda2=penalty, direction_d=d,
              ols_variance=sigma2 / (n * d), ridge=rows,
              equal_norm_direction_prediction_mse=prediction_mse)

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
