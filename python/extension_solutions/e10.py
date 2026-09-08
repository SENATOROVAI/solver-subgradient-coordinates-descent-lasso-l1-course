"""Ответ Д10.3: независимые обучения в новых единицах."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pure_core import marketing_data
X, y, _ = marketing_data(n=120, seed=824)
X, y = np.array(X), np.array(y)
U = X.copy()
U[:, 1] *= 100

def fit(A, target, scale):
    model = Lasso(alpha=.3, max_iter=20000, tol=1e-10)
    if scale:
        model = make_pipeline(StandardScaler(), model)
    return model.fit(A, target)

changes = []
for scale in [False, True]:
    a, b = fit(X, y, scale), fit(U, y, scale)
    delta = float(np.max(abs(a.predict(X) - b.predict(U))))
    c = fit(X, y + 100, scale)
    assert np.max(abs(c.predict(X) - a.predict(X) - 100)) < 1e-8
    if scale:
        assert delta < 1e-8
    changes.append(delta)
s = np.sqrt(.04 * .96)
RESULT = {'raw_and_scaled_prediction_change': changes,
          'binary_scale': float(s),
          'binary_values': [float(-.04 / s), float(.96 / s)]}
print(RESULT)
