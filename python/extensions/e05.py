"""Д05. Точный порог влияния далёкой строки на MAE."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor, LinearRegression
from pure_core import mae, mse
from plot_utils import savefig
from extra_utils import finish_extra

values = [0., 0., 6.]
assert mse(values, [2.] * 3) == 8.
assert mse(values, [3.] * 3) == 9.
assert mae(values, [0.] * 3) == 2.

# Чистый Python: одна прямая без свободного члена.
slopes = [-1.5 + k / 200 for k in range(601)]
fig, ax = plt.subplots(figsize=(7.4, 3.8))
for M in [3, 6, 10]:
    values = [6 * abs(1 - w) + M * abs(1 + w) for w in slopes]
    ax.plot(slopes, values, label=f'M = {M}')
ax.set(xlabel='Наклон w', ylabel='Сумма абсолютных ошибок')
ax.legend()
savefig(fig, 'e05_leverage_threshold')

# Библиотечная проверка и независимые будущие строки.
rng = random.Random(826)
xt = [rng.uniform(1, 3) for _ in range(1000)]
y_clean = [x + rng.gauss(0, .1) for x in xt]
y_shift = [-x + rng.gauss(0, .1) for x in xt]
records = []
for M in [3, 6, 10]:
    X, y = np.array([[1.], [2.], [3.], [float(M)]]), [1., 2., 3., -M]
    lad = QuantileRegressor(alpha=0, quantile=.5,
                           fit_intercept=False).fit(X, y)
    ols = LinearRegression(fit_intercept=False).fit(X, y)
    expected_ols = (14 - M * M) / (14 + M * M)
    assert abs(ols.coef_[0] - expected_ols) < 1e-12
    assert abs(lad.coef_[0] - (1 if M < 6 else -1)) < 1e-9 or M == 6
    if M == 6:
        assert -1 - 1e-10 <= lad.coef_[0] <= 1 + 1e-10
        assert abs(np.mean(abs(y - lad.predict(X))) - 3) < 1e-10
    prediction = [float(lad.coef_[0]) * x for x in xt]
    records.append({'M': M, 'lad_slope': float(lad.coef_[0]),
                    'ols_slope': float(ols.coef_[0]),
                    'future_clean_mae': mae(y_clean, prediction),
                    'future_reversed_mae': mae(y_shift, prediction)})
fig, ax = plt.subplots(figsize=(7.4, 3.8))
pos = np.arange(3)
ax.bar(pos - .18, [v['future_clean_mae'] for v in records], .36,
       label='Будущее: прежняя связь y ≈ x')
ax.bar(pos + .18, [v['future_reversed_mae'] for v in records], .36,
       label='Будущее: новая связь y ≈ −x')
ax.set_xticks(pos, ['M=3', 'M=6', 'M=10'])
ax.set(xlabel='Далёкая обучающая строка (M, −M)', ylabel='MAE на новых строках')
ax.legend()
savefig(fig, 'e05_future_distribution')
RESULT = {'experiments': records, 'flat_interval_at_M6': [-1., 1.],
          'test_rows': len(xt), 'seed': 826}
FIGURES = ['e05_leverage_threshold', 'e05_future_distribution']
finish_extra('e05', RESULT)
