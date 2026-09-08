"""Д26.3. Отдельные фиксированные и заново подобранные модели."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import random
import numpy as np  # Явный мост к библиотечному повторному подбору.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from pure_core import marketing_data, predict
from geometry_extra import stability_metrics
from extra_utils import finish_extra

X, y, _ = marketing_data(n=60, seed=822)
X, y = np.array(X), np.array(y)
rng = random.Random(822)
records = {'fixed': [], 'retuned': []}
settings, predictions = [], {'fixed': [], 'retuned': []}
for repeat in range(12):
    ix = rng.sample(range(60), 48)
    def make_pipe():
        return Pipeline([('scale', StandardScaler()),
                         ('model', ElasticNet(alpha=.24, l1_ratio=.5,
                                              tol=1e-9,
                                              max_iter=30000))])
    fixed = make_pipe().fit(X[ix], y[ix])
    search = GridSearchCV(make_pipe(),
                         {'model__alpha': [.06, .2],
                          'model__l1_ratio': [.5, 1.]},
                         cv=KFold(3, shuffle=True, random_state=822),
                         scoring='neg_mean_squared_error').fit(X[ix], y[ix])
    settings.append(search.best_params_)
    for name, pipe in [('fixed', fixed),
                       ('retuned', search.best_estimator_)]:
        scale, model = pipe.named_steps['scale'], pipe.named_steps['model']
        w = model.coef_ / scale.scale_
        b = float(model.intercept_ - w @ scale.mean_)
        records[name].append(w.tolist())
        predictions[name].append(predict(X[:5], w, b))
        assert np.max(np.abs(pipe.predict(X[:5])
                              - predictions[name][-1])) < 1e-9
reports = {name: {'exact': stability_metrics(ws, {0, 4}, 0.),
                  'epsilon': stability_metrics(ws, {0, 4}, 1e-8)}
           for name, ws in records.items()}
never = stability_metrics([[1., 0.], [0., 0.]], {0})
assert never['positive_if_selected'][1] is None
toy = stability_metrics([[1., 0., 0.], [0., 1., 0.],
                         [1., 0., 1.], [0., 1., 0.]], {0, 1})
assert abs(toy['mean_jaccard'] - .25) < 1e-12
RESULT = {'repetitions': 12, 'descriptive_only': True,
          'reports': reports, 'settings': settings,
          'common_predictions': predictions, 'toy': toy,
          'never_selected': never, 'manual_mc_se': .05}
finish_extra('solution_e26', RESULT)
