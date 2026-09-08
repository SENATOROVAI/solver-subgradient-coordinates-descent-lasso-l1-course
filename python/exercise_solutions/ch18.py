"""18.3. Девять комбинаций, три сгиба, полный Pipeline."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import marketing_data

# Эта часть намеренно использует библиотечный мост.
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y, _ = marketing_data(n=90, seed=18)
folds = KFold(n_splits=3, shuffle=True, random_state=18)
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", ElasticNet(max_iter=30000, tol=1e-8)),
])
grid = {
    "model__alpha": [0.03, 0.15, 0.6],
    "model__l1_ratio": [0.2, 0.6, 1.0],
}
search = GridSearchCV(
    pipe, grid, cv=folds, scoring="neg_mean_squared_error",
    error_score="raise", refit=True,
)
search.fit(X, y)
assert len(search.cv_results_["params"]) == 9
assert search.n_splits_ == 3
scaler = search.best_estimator_.named_steps["scale"]
assert int(scaler.n_samples_seen_) == len(y)
alpha = float(search.best_params_["model__alpha"])
rho = float(search.best_params_["model__l1_ratio"])
best_mse = -float(search.best_score_)
assert math.isfinite(best_mse) and best_mse >= 0.0
rows = []
for params, score in zip(
    search.cv_results_["params"],
    search.cv_results_["mean_test_score"],
):
    rows.append({
        "alpha": float(params["model__alpha"]),
        "l1_ratio": float(params["model__l1_ratio"]),
        "cv_mse": -float(score),
    })

RESULT = {
    "seed": 18, "n": len(y), "candidates": rows,
    "best_cv_mse": best_mse, "alpha": alpha, "l1_ratio": rho,
    "lambda1": alpha * rho, "lambda2": alpha * (1.0 - rho),
}
if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
