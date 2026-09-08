"""20.3. Пять семейств; выбор по CV до обращения к тесту."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import mae, marketing_data, mse

# Библиотечный мост: NumPy и scikit-learn.
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y, true_coef = marketing_data(seed=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
folds = list(KFold(
    n_splits=5, shuffle=True, random_state=42
).split(X_train))
alphas = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
families = [
    ("Среднее", DummyRegressor(strategy="mean"), {}),
    ("OLS", LinearRegression(), {}),
    ("Lasso", Lasso(max_iter=30000, tol=1e-10),
     {"model__alpha": alphas}),
    ("Ridge", Ridge(),
     {"model__alpha": [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]}),
    ("ElasticNet", ElasticNet(max_iter=30000, tol=1e-10),
     {"model__alpha": alphas,
      "model__l1_ratio": [0.1, 0.3, 0.5, 0.8, 0.95]}),
]
models = {}
rows = []
for name, estimator, grid in families:
    pipe = Pipeline([
        ("scale", StandardScaler()), ("model", estimator)
    ])
    search = GridSearchCV(
        pipe, grid, cv=folds, scoring="neg_mean_squared_error",
        refit=True, error_score="raise",
    ).fit(X_train, y_train)
    models[name] = search.best_estimator_
    rows.append({
        "family": name, "cv_mse": -float(search.best_score_),
        "parameters": search.best_params_,
    })

# Выбор семейства завершён до вычисления любой тестовой ошибки.
winner = min(rows, key=lambda row: row["cv_mse"])["family"]
chosen = models[winner]
assert len(rows) == 5
assert len(y_train) == 120 and len(y_test) == 40
for row in rows:
    prediction = models[row["family"]].predict(X_test).tolist()
    row["test_mae"] = mae(y_test, prediction)
    row["test_mse"] = mse(y_test, prediction)
    assert math.isfinite(row["test_mse"])


def original_units(pipe):
    """Возвращает коэффициенты и свободный член в исходных единицах."""
    estimator = pipe.named_steps["model"]
    scaler = pipe.named_steps["scale"]
    if isinstance(estimator, DummyRegressor):
        return [0.0] * len(X[0]), float(estimator.constant_[0, 0])
    w = [float(a / b) for a, b in zip(
        estimator.coef_, scaler.scale_
    )]
    b = float(estimator.intercept_) - sum(
        value * mean for value, mean in zip(w, scaler.mean_)
    )
    return w, b


# Проверяем пересчёт на тренировочных строках.
raw_coef, raw_intercept = original_units(chosen)
manual_prediction = [
    raw_intercept + sum(a * b for a, b in zip(row, raw_coef))
    for row in X_train
]
gap = max(abs(a - b) for a, b in zip(
    manual_prediction, chosen.predict(X_train)
))
assert gap < 1e-8

# 100 подвыборок без возвращения, только из тренировочной части.
rng = np.random.default_rng(42)
threshold = 1e-8
repetitions = 100
sample_size = int(0.8 * len(X_train))
weight_samples = []
for _ in range(repetitions):
    indices = rng.choice(
        len(X_train), size=sample_size, replace=False
    )
    sample_x = [X_train[i] for i in indices]
    sample_y = [y_train[i] for i in indices]
    fitted = clone(chosen).fit(sample_x, sample_y)
    weights, _ = original_units(fitted)
    weight_samples.append(weights)
frequencies = [
    sum(abs(row[j]) > threshold for row in weight_samples)
    / repetitions for j in range(len(X[0]))
]
assert all(0.0 <= value <= 1.0 for value in frequencies)
assert winner == min(rows, key=lambda row: row["cv_mse"])["family"]

RESULT = {
    "seed": 42, "train_n": len(y_train), "test_n": len(y_test),
    "selected_family_before_test": winner, "comparison": rows,
    "raw_coef": raw_coef, "raw_intercept": raw_intercept,
    "true_coef_for_diagnosis_only": true_coef,
    "coefficient_roundtrip_gap": float(gap),
    "stability_repetitions": repetitions,
    "stability_training_rows_per_fit": sample_size,
    "nonzero_threshold": threshold,
    "selection_frequency": frequencies,
    "raw_weight_samples": weight_samples,
}
if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
