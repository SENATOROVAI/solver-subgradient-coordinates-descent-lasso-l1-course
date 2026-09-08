"""15.3. Проверка нормировки Ridge и повторения строк."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import coordinate_descent, predict

# Библиотечный мост для независимой проверки.
from sklearn.linear_model import Ridge

X = [[a, b] for a in [-1.0, 1.0] for b in [-1.0, 1.0]]
y = [2.0 * row[0] + 0.5 * row[1] for row in X]
lambda2 = 0.5
rows = []
original_coef = None
for repeats in [1, 3]:
    data = X * repeats
    target = y * repeats
    model = coordinate_descent(
        data, target, alpha=lambda2, l1_ratio=0.0, tol=1e-11
    )
    assert model["converged"]
    alpha_ridge = len(target) * lambda2
    reference = Ridge(alpha=alpha_ridge).fit(data, target)
    actual = predict(X, model["coef"], model["intercept"])
    expected = reference.predict(X)
    difference = max(abs(a - b) for a, b in zip(actual, expected))
    assert difference < 1e-7
    if original_coef is None:
        original_coef = model["coef"][:]
    else:
        assert max(abs(a - b) for a, b in zip(
            original_coef, model["coef"]
        )) < 1e-7
    assert abs(model["coef"][0] - 4.0 / 3.0) < 1e-7
    assert abs(model["coef"][1] - 1.0 / 3.0) < 1e-7
    rows.append({
        "n": len(target), "alpha_R": alpha_ridge,
        "coef": model["coef"], "max_prediction_gap": difference,
    })

RESULT = {"lambda2": lambda2, "comparisons": rows}
if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
