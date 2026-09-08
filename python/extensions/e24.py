"""Д24. Группы, время и нелинейные остатки."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GroupKFold, KFold, TimeSeriesSplit, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from plot_utils import savefig
from extra_utils import finish_extra

# Разбиения отвечают разным вопросам о новых наблюдениях.
Xsmall = np.arange(24).reshape(-1, 1)
groups = np.repeat(np.arange(6), 4)
splitters = [KFold(3, shuffle=True, random_state=825),
             GroupKFold(3), TimeSeriesSplit(3, gap=1)]
labels = ['Случайные строки', 'Новые группы', 'Будущие моменты']
fig, axes = plt.subplots(3, 1, figsize=(8., 6.3), sharex=True)
intersections = []
for ax, splitter, label in zip(axes, splitters, labels):
    codes = np.zeros((3, 24))
    overlap = []
    pairs = (splitter.split(Xsmall, groups=groups)
             if isinstance(splitter, GroupKFold)
             else splitter.split(Xsmall))
    for i, (train, valid) in enumerate(pairs):
        codes[i, train], codes[i, valid] = 1, 2
        overlap.append(len(set(groups[train]) & set(groups[valid])))
        if isinstance(splitter, TimeSeriesSplit):
            assert train.max() + 1 < valid.min()
    if isinstance(splitter, GroupKFold):
        assert overlap == [0, 0, 0]
    intersections.append(overlap)
    ax.imshow(codes, aspect='auto', interpolation='nearest', vmin=0,
              vmax=2, cmap=ListedColormap(['white', '#24577C', '#CE7334']))
    ax.set(yticks=[0, 1, 2], yticklabels=['1', '2', '3'],
           ylabel='Фолд', title=label)
axes[-1].set_xlabel('Номер строки; синий: обучение, оранжевый: проверка')
savefig(fig, 'e24_split_choices')

# Свежий независимый опыт: тест отделён до выбора степени.
rng = np.random.default_rng(825)
X = rng.uniform(-2, 2, size=(180, 1))
y = 2 + X[:, 0] + 2 * X[:, 0] ** 2 + rng.normal(0, .4, 180)
Xt = rng.uniform(-2, 2, size=(600, 1))
yt = 2 + Xt[:, 0] + 2 * Xt[:, 0] ** 2 + rng.normal(0, .4, 600)
pipe = Pipeline([('poly', PolynomialFeatures(include_bias=False)),
                 ('scale', StandardScaler()), ('ridge', Ridge())])
cv = KFold(5, shuffle=True, random_state=825)
search = GridSearchCV(pipe, {'poly__degree': [1, 2, 3],
                            'ridge__alpha': [.01, .1, 1., 10.]},
                      cv=cv, scoring='neg_mean_squared_error')
search.fit(X, y)
linear = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=.1))])
linear.fit(X, y)
fig, axes = plt.subplots(1, 2, figsize=(9., 4.2))
for ax, fitted, label in [(axes[0], linear, 'Только прямая'),
                           (axes[1], search.best_estimator_, 'Выбор внутри CV')]:
    forecast = fitted.predict(X)
    ax.scatter(X[:, 0], y - forecast, s=14, alpha=.65)
    ax.axhline(0, color='black', ls='--')
    ax.set(xlabel='Исходный признак x', ylabel='Остаток y − прогноз',
           title=label)
savefig(fig, 'e24_nonlinear_residuals')
RESULT = {'group_overlap_by_splitter': intersections,
          'best_params': search.best_params_,
          'cv_mse': float(-search.best_score_),
          'linear_test_mse': float(mean_squared_error(yt, linear.predict(Xt))),
          'selected_test_mse': float(mean_squared_error(yt, search.predict(Xt))),
          'test_rows': len(yt)}
FIGURES = ['e24_split_choices', 'e24_nonlinear_residuals']
finish_extra('e24', RESULT)
