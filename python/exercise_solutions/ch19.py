"""19.3. Подбор только на train, итоговая оценка на test."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import mae, marketing_data, mse

# Библиотечный мост для конвейера и кросс-валидации.
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y, _ = marketing_data(n=160, seed=19)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=19
)
folds = KFold(n_splits=4, shuffle=True, random_state=19)
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", ElasticNet(max_iter=30000, tol=1e-8)),
])
grid = {
    "model__alpha": [0.01, 0.05, 0.2],
    "model__l1_ratio": [0.2, 0.6, 1.0],
}
search = GridSearchCV(
    pipe, grid, cv=folds, scoring="neg_mean_squared_error",
    error_score="raise", refit=True,
)
search.fit(X_train, y_train)
chosen = search.best_estimator_
scaler = chosen.named_steps["scale"]
assert int(scaler.n_samples_seen_) == len(y_train)
means = [
    sum(row[j] for row in X_train) / len(X_train)
    for j in range(len(X_train[0]))
]
assert max(abs(a - b) for a, b in zip(means, scaler.mean_)) < 1e-10
# Обращение к тесту происходит после выбора параметров.
prediction = chosen.predict(X_test).tolist()
RESULT = {
    "seed": 19, "train_n": len(y_train), "test_n": len(y_test),
    "selected_parameters": search.best_params_,
    "cv_mse": -float(search.best_score_),
    "test_mae": mae(y_test, prediction),
    "test_mse": mse(y_test, prediction),
    "scaler_training_rows": int(scaler.n_samples_seen_),
    "training_means": means,
}
assert RESULT["train_n"] == 120 and RESULT["test_n"] == 40
if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
