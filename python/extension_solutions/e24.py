"""Ответ Д24.3: проверяем разбиение до подбора модели."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.model_selection import (
    GroupKFold, TimeSeriesSplit, KFold, GridSearchCV)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
rng = np.random.default_rng(925)
X = rng.uniform(-2, 2, (180, 1))
y = 2 + X[:, 0] + 2 * X[:, 0] ** 2 + rng.normal(0, .4, 180)
groups = np.repeat(np.arange(30), 6)
for train, valid in GroupKFold(5).split(X, y, groups):
    assert not set(groups[train]) & set(groups[valid])
# Только проверка порядка индексов, не CV этих неупорядоченных x.
for train, valid in TimeSeriesSplit(5, gap=2).split(X):
    assert train.max() + 2 < valid.min()
pipe = make_pipeline(PolynomialFeatures(include_bias=False),
                     StandardScaler(), Ridge())
search = GridSearchCV(pipe, {
    'polynomialfeatures__degree': [1, 2, 3, 4],
    'ridge__alpha': [.01, .1, 1., 10.]},
    cv=KFold(5, shuffle=True, random_state=925),
    scoring='neg_mean_squared_error').fit(X, y)
RESULT = {'best_params': search.best_params_,
          'cv_mse': float(-search.best_score_),
          'group_and_time_checks': 'passed'}
print(RESULT)
