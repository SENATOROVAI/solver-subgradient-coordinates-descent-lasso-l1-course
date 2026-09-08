"""Ответ Д01.3: сетка вокруг иррационального наклона."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import numpy as np
from pure_core import grid_mae
X = [[-2.], [-1.], [0.], [1.], [2.]]
y = [math.sqrt(2) * row[0] + .37 for row in X]
records = []
for h in [.5, .25, .1]:
    ws, bs = np.arange(0, 2 + h / 2, h), np.arange(0, 1 + h / 2, h)
    fit = grid_mae(X, y, ws.tolist(), bs.tolist())
    loss = abs(np.array(y) - ws[:, None, None] * np.array(X)[:, 0]
               - bs[None, :, None]).mean(axis=2)
    assert np.allclose(loss.ravel(), [v[2] for v in fit['records']])
    assert fit['loss'] <= 1.1 * h + 1e-12
    records.append({'step': h, 'loss': fit['loss'], 'bound': 1.1 * h})
RESULT = {'grid': records, 'manual_mae': 1., 'manual_coverage': .5,
          'manual_bound': .5}
print(RESULT)
