"""12.3: проверка настоящей координатной траектории и остатка."""
from pathlib import Path
import json
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import lasso_cd, predict, residuals

rng = random.Random(12)
X = []
for _ in range(24):
    first = rng.gauss(0.0, 1.0)
    second = 0.8 * first + 0.6 * rng.gauss(0.0, 1.0)
    X.append([first, second])
y = [2.0 * a - 0.7 * b for a, b in X]
model = lasso_cd(X, y, alpha=0.1, fit_intercept=False,
                 keep_steps=True, tol=1e-10, max_iter=10000)
trace = model["trace"]
assert model["converged"]
assert all(b["objective"] <= a["objective"] + 1e-10
           for a, b in zip(trace, trace[1:]))
r = y[:]
for before, after in zip(trace, trace[1:]):
    j = after["coordinate"]
    assert all(abs(a - b) < 1e-12 for k, (a, b)
               in enumerate(zip(before["coef"], after["coef"]))
               if k != j)
    change = after["coef"][j] - before["coef"][j]
    r = [value - row[j] * change for value, row in zip(r, X)]
direct = residuals(y, predict(X, model["coef"]))
gap = max(abs(a - b) for a, b in zip(r, direct))
assert gap < 1e-10
RESULT = {"coef": model["coef"], "objective": model["objective"],
          "epochs": model["n_iter"], "coordinate_steps": len(trace) - 1,
          "max_residual_gap": gap, "converged": model["converged"]}

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
