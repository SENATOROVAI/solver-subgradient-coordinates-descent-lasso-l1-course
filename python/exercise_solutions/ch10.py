"""10.3: независимый перебор внутри ромба с шагом 0.02."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import lasso_cd, mse, predict

X = [[a, b] for a in [-1.0, 1.0] for b in [-1.0, 1.0]]
y = [2.0 * a + 0.5 * b for a, b in X]
best = None
for i in range(-60, 61):
    for j in range(-60, 61):
        w = [i / 50.0, j / 50.0]
        if sum(abs(value) for value in w) > 1.2 + 1e-12:
            continue
        error = mse(y, predict(X, w)) / 2.0
        if best is None or error < best["half_mse"]:
            best = {"coef": w, "half_mse": error}
model = lasso_cd(X, y, alpha=0.8, fit_intercept=False)
max_gap = max(abs(a - b) for a, b in zip(best["coef"], model["coef"]))
assert max_gap < 1e-10
assert abs(best["half_mse"] - 0.445) < 1e-10
assert abs(sum(abs(v) for v in best["coef"]) - 1.2) < 1e-10
RESULT = {"grid": best, "lasso_coef": model["coef"],
          "max_weight_gap": max_gap, "grid_step": 0.02}

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
