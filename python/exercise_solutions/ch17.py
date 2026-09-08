"""17.3. Обычный ElasticNet и три предельных режима."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import coordinate_descent, predict

# Библиотечный мост; наше ядро библиотеку не вызывает.
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.linear_model import LinearRegression, Ridge

X = [[a, b] for a in [-1.0, 1.0] for b in [-1.0, 1.0]]
y = [5.0 + 2.0 * row[0] + 0.5 * row[1] for row in X]
cases = [
    ("ElasticNet", 0.2, 0.4, ElasticNet(
        alpha=0.2, l1_ratio=0.4, tol=1e-11, max_iter=10000
    )),
    ("Lasso", 0.2, 1.0, Lasso(
        alpha=0.2, tol=1e-11, max_iter=10000
    )),
    ("Ridge", 0.2, 0.0, Ridge(alpha=len(y) * 0.2)),
    ("OLS", 0.0, 0.4, LinearRegression()),
]
rows = []
for name, alpha, rho, reference in cases:
    model = coordinate_descent(
        X, y, alpha=alpha, l1_ratio=rho, tol=1e-11
    )
    assert model["converged"]
    reference.fit(X, y)
    actual = predict(X, model["coef"], model["intercept"])
    expected = reference.predict(X)
    difference = max(abs(a - b) for a, b in zip(actual, expected))
    assert difference < 1e-7
    assert abs(model["intercept"] - 5.0) < 1e-7
    rows.append({
        "regime": name, "coef": model["coef"],
        "intercept": model["intercept"],
        "max_prediction_gap": difference,
        "converged": model["converged"],
    })

RESULT = {"comparisons": rows, "tolerance": 1e-7}
if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
