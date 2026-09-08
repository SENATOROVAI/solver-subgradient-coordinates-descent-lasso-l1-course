"""13.3: бюджет, нулевые столбцы и обучение без штрафа."""
from pathlib import Path
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import lasso_cd, predict

X = [[(i - 14.5) / 10.0,
      (i - 14.5) / 10.0 + 0.05 * math.sin(i)] for i in range(30)]
y = [1.7 * a + 0.5 * b for a, b in X]
short = lasso_cd(X, y, alpha=0.1, max_iter=1, tol=1e-10)
long = lasso_cd(X, y, alpha=0.1, max_iter=50000, tol=1e-10)
assert not short["converged"] and short["status"] == "max_iter"
assert long["converged"] and long["status"] == "converged"
assert long["objective"] <= short["objective"] + 1e-10
assert all(b <= a + 1e-10
           for a, b in zip(long["history"], long["history"][1:]))
Z = [[a, b, 0.0] for a in [-1.0, 1.0] for b in [-1.0, 1.0]]
target = [7.0 + 2.0 * a - 0.5 * b for a, b, _ in Z]
unpenalized = lasso_cd(Z, target, alpha=0.0, tol=1e-12)
prediction = predict(Z, unpenalized["coef"],
                     unpenalized["intercept"])
assert unpenalized["converged"]
assert abs(unpenalized["coef"][-1]) < 1e-12
assert max(abs(a - b) for a, b in zip(prediction, target)) < 1e-10
all_zero = lasso_cd(Z, target, alpha=100.0)
assert all(abs(w) < 1e-12 for w in all_zero["coef"])
RESULT = {
    "short_status": short["status"], "long_status": long["status"],
    "long_epochs": long["n_iter"],
    "short_objective": short["objective"],
    "long_objective": long["objective"],
    "alpha_zero_coef": unpenalized["coef"],
    "alpha_zero_intercept": unpenalized["intercept"],
    "large_alpha_coef": all_zero["coef"],
}

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
