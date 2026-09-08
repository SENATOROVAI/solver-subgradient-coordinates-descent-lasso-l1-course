"""16.3. Только L2 меняется; вычисления на обычных списках."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import elasticnet_cd, mse, predict

X = [[-1.0, -1.0], [1.0, 1.0]]
y = [-3.0, 3.0]
lambda1 = 0.2
rows = []
previous_sum = float("inf")
for lambda2 in [0.0, 0.05, 0.2, 0.8, 2.0]:
    alpha = lambda1 + lambda2
    rho = lambda1 / alpha
    assert abs(alpha * rho - lambda1) < 1e-12
    assert abs(alpha * (1.0 - rho) - lambda2) < 1e-12
    model = elasticnet_cd(
        X, y, alpha=alpha, l1_ratio=rho,
        max_iter=20000, tol=1e-11,
    )
    assert model["converged"]
    w = model["coef"]
    if lambda2 > 0.0:
        expected = 2.8 / (2.0 + lambda2)
        assert max(abs(value - expected) for value in w) < 1e-8
        assert abs(w[0] - w[1]) < 1e-8
    else:
        assert abs(sum(w) - 2.8) < 1e-8
    assert sum(w) <= previous_sum + 1e-8
    previous_sum = sum(w)
    prediction = predict(X, w, model["intercept"])
    rows.append({
        "lambda1": alpha * rho, "lambda2": lambda2,
        "alpha": alpha, "l1_ratio": rho, "coef": w,
        "sum_coef": sum(w), "mse": mse(y, prediction),
        "objective": model["objective"],
    })

RESULT = {"fixed_lambda1": lambda1, "experiments": rows}
if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
