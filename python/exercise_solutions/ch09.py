"""09.3: путь Lasso на четырёх ортогональных строках."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import lasso_cd, mse, predict

X = [[a, b] for a in [-1.0, 1.0] for b in [-1.0, 1.0]]
y = [2.0 * a + 0.5 * b for a, b in X]
alphas = [0.1, 0.5, 1.0, 2.0]
expected = [[1.9, 0.4], [1.5, 0.0], [1.0, 0.0], [0.0, 0.0]]
records = []
for alpha, target in zip(alphas, expected):
    model = lasso_cd(X, y, alpha=alpha, fit_intercept=False)
    w = model["coef"]
    assert model["converged"]
    assert max(abs(a - b) for a, b in zip(w, target)) < 1e-10
    error = mse(y, predict(X, w))
    penalty = alpha * sum(abs(value) for value in w)
    records.append({"alpha": alpha, "coef": w, "mse": error,
                    "penalty": penalty,
                    "zeros": sum(abs(value) < 1e-10 for value in w)})
assert all(a["mse"] <= b["mse"] + 1e-12
           for a, b in zip(records, records[1:]))
RESULT = {"records": records}

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
