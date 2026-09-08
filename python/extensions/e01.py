"""Д01. Одинаковая MAE и точность сеточного поиска."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from pure_core import mae, grid_mae, predict
from plot_utils import savefig
from extra_utils import finish_extra

# Чистый Python: одно среднее скрывает расположение ошибок.
errors = [[1., 1., 1., 1.], [0., 0., 0., 4.]]
metrics = [{"mae": sum(v) / len(v), "max": max(v),
            "within_1": sum(x <= 1 for x in v) / len(v)}
           for v in errors]
y = [1., 2., 4.]
p = [0., 3., 2.]
assert mae(y, p) == mae([v + 7 for v in y], [v + 7 for v in p])
assert mae(y, p) * 3 == mae([3 * v for v in y], [3 * v for v in p])
assert mae(y, p) == mae(y * 2, p * 2)
fig, ax = plt.subplots(figsize=(7.4, 3.8))
for label, values in zip(['A: все ошибки равны 1', 'B: один промах 4'],
                         errors):
    sx = [-.1] + sorted(values) + [4.3]
    sy = [0.] + [(i + 1) / len(values) for i in range(len(values))] + [1.]
    ax.step(sx, sy, where='post', label=label)
ax.set(xlabel='Порог абсолютной ошибки', ylabel='Доля ошибок не выше порога')
ax.legend(loc='lower right')
savefig(fig, 'e01_error_cdf')

# Одинаковые оси сетки: строка отвечает w, столбец отвечает b.
X = [[-2.], [-1.], [0.], [1.], [2.]]
y = [1.3 * row[0] + .7 for row in X]
widths = [.5, .25, .1]
records = []
fig, axes = plt.subplots(1, 2, figsize=(9., 4.1))
for h in widths:
    ws = np.arange(0., 2. + h / 2, h)
    bs = np.arange(0., 1. + h / 2, h)
    fit = grid_mae(X, y, ws.tolist(), bs.tolist())
    # NumPy: временный массив имеет форму (len(ws),len(bs),n).
    residual = (np.array(y)[None, None, :] -
                ws[:, None, None] * np.array(X)[:, 0] -
                bs[None, :, None])
    losses = np.abs(residual).mean(axis=2)
    loop = np.array([v[2] for v in fit['records']]).reshape(losses.shape)
    assert np.allclose(loop, losses)
    bound = h / 2 * (sum(abs(row[0]) for row in X) / len(X) + 1)
    assert fit['loss'] <= bound + 1e-12
    records.append({'step': h, 'loss': fit['loss'], 'bound': bound,
                    'candidates': len(ws) * len(bs),
                    'one_array_bytes': residual.nbytes})
    if h == .25:
        z = axes[0].contourf(bs, ws, losses, levels=14)
        fig.colorbar(z, ax=axes[0], label='MAE')
        axes[0].plot(.7, 1.3, 'w*', ms=13, label='Точный минимум')
        axes[0].legend()
axes[0].set(xlabel='Свободный член b', ylabel='Наклон w')
axes[1].plot(widths, [v['loss'] for v in records], 'o-', label='Избыток MAE')
axes[1].plot(widths, [v['bound'] for v in records], 's--', label='Верхняя оценка')
axes[1].set(xlabel='Шаг сетки', ylabel='Ошибка относительно минимума 0')
axes[1].legend()
savefig(fig, 'e01_grid_accuracy')
# Ошибка broadcasting видна по форме ещё до вычисления среднего.
a = np.array([1., 2., 3.])
wrong_shape = (a[:, None] - a).shape
assert wrong_shape == (3, 3)
RESULT = {'error_profiles': metrics, 'grid': records,
          'wrong_broadcast_shape': list(wrong_shape),
          'invariants_passed': 3}
FIGURES = ['e01_error_cdf', 'e01_grid_accuracy']
finish_extra('e01', RESULT)
