"""08.3: два равных способа вычислить прогноз после стандартизации."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import (lasso_cd, marketing_data, predict, scale_fit,
                       scale_transform, unscale_coefficients)

X, y, _ = marketing_data(seed=42)
X = [row + [7.0] for row in X]
# Здесь отдельное упражнение: первые 120 строк идут в обучение.
train, test = X[:120], X[120:]
stats = scale_fit(train)
U_train = scale_transform(train, stats)
U_test = scale_transform(test, stats)
model = lasso_cd(U_train, y[:120], alpha=0.1, tol=1e-10)
w, b = unscale_coefficients(
    model["coef"], model["intercept"], stats
)
standard = predict(U_test, model["coef"], model["intercept"])
original = predict(test, w, b)
max_gap = max(abs(a - c) for a, c in zip(standard, original))
assert model["converged"]
assert max_gap < 1e-10
assert abs(stats["scale"][-1] - 1.0) < 1e-12
assert abs(model["coef"][-1]) < 1e-12
assert all(abs(row[-1]) < 1e-12 for row in U_train)
RESULT = {
    "max_prediction_gap": max_gap,
    "constant_scale": stats["scale"][-1],
    "constant_weight": model["coef"][-1],
    "original_intercept": b,
    "n_iter": model["n_iter"],
}

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
