"""Ответ Д05.3: сумма |x| задаёт порог."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import QuantileRegressor
records = []
for M in [5., 7., 9.]:
    X = np.array([[1.], [2.], [4.], [M]])
    y = np.array([1., 2., 4., -M])
    ws = np.linspace(-1.5, 1.5, 601)
    losses = abs(y[None, :] - ws[:, None] * X[:, 0]).mean(axis=1)
    model = QuantileRegressor(alpha=0, fit_intercept=False).fit(X, y)
    actual = float(abs(y - model.predict(X)).mean())
    assert abs(actual - losses.min()) < 1e-10
    positive = QuantileRegressor(alpha=0, fit_intercept=False).fit(
        X, np.array([1., 2., 4., M]))
    assert abs(positive.coef_[0] - 1) < 1e-10
    records.append({'M': M, 'slope': float(model.coef_[0]), 'mae': actual})
RESULT = {'threshold': 7., 'threshold_mae': 3.5, 'fits': records}
print(RESULT)
