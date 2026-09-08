"""Д26. Устойчивость группы и всей процедуры выбора модели."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import random
from statistics import mean, pstdev
from collections import Counter
import numpy as np  # Явный мост к sklearn и рисункам, не часть метрик.
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from pure_core import marketing_data, coordinate_descent, predict
from geometry_extra import stability_metrics, sum_variance
from plot_utils import savefig
from extra_utils import finish_extra


def pipeline(alpha=.12, rho=1.):
    return Pipeline([('scale', StandardScaler()),
                     ('model', ElasticNet(alpha=alpha, l1_ratio=rho,
                                          tol=1e-9, max_iter=30000))])


def original_parameters(pipe):
    scale, model = pipe.named_steps['scale'], pipe.named_steps['model']
    w = model.coef_ / scale.scale_
    b = float(model.intercept_ - w @ scale.mean_)
    return w.tolist(), b


def search_model():
    return GridSearchCV(pipeline(),
                        {'model__alpha': [.06, .2],
                         'model__l1_ratio': [.5, 1.]},
                        cv=KFold(3, shuffle=True, random_state=715),
                        scoring='neg_mean_squared_error', n_jobs=1)


# Отдельный обучающий опыт: финальный тест основного проекта не открыт.
X, y, truth = marketing_data(n=120, seed=715)
probe, _, _ = marketing_data(n=60, seed=716)
X, y = np.array(X), np.array(y)
B, size = 60, 96
rng = random.Random(715)
weights = {name: [] for name in ['Lasso', 'ElasticNet', 'С подбором']}
forecasts = {name: [] for name in weights}
settings, subsets = [], []
for repeat in range(B):
    subset = rng.sample(range(len(y)), size)
    subsets.append(subset)
    a, target = X[subset], y[subset]
    candidates = {'Lasso': pipeline(.12, 1.).fit(a, target),
                  'ElasticNet': pipeline(.24, .5).fit(a, target)}
    search = search_model().fit(a, target)
    candidates['С подбором'] = search.best_estimator_
    settings.append(search.best_params_)
    for name, pipe in candidates.items():
        w, b = original_parameters(pipe)
        weights[name].append(w)
        forecasts[name].append(predict(probe, w, b))
        assert np.max(np.abs(pipe.predict(probe)
                              - forecasts[name][-1])) < 1e-9
metrics = {name: stability_metrics(ws, {0, 4}, 1e-8)
           for name, ws in weights.items()}
exact = {name: stability_metrics(ws, {0, 4}, 0.)
         for name, ws in weights.items()}
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.7))
locations = np.arange(8)
for offset, (name, result) in zip([-.25, 0., .25], metrics.items()):
    axes[0].bar(locations + offset, result['frequency'], width=.25,
                label=name)
axes[0].set(xticks=locations, xticklabels=[f'x{i + 1}' for i in locations],
            ylabel='Доля выбранных в 60 подвыборках', ylim=(0, 1.1))
axes[0].legend(fontsize=11, loc='upper center',
                bbox_to_anchor=(.5, 1.22), ncol=3)
for offset, (name, result) in zip([-.25, 0., .25], metrics.items()):
    axes[1].bar(np.arange(3) + offset,
                [result['group_any'], result['group_all'],
                 result['mean_jaccard']], width=.25, label=name)
axes[1].set(xticks=range(3), xticklabels=['Хотя бы\nодин', 'Оба', 'Жаккар'],
            ylabel='Устойчивость набора', ylim=(0, 1.1),
            title='Группа x1 и x5 задана заранее')
savefig(fig, 'e26_selection')

fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.7))
signs = [[np.nan if value is None else value
          for value in row['positive_if_selected']]
         for row in metrics.values()]
axes[0].imshow(signs, vmin=0., vmax=1., cmap='Blues', aspect='auto')
for i, row in enumerate(signs):
    for j, value in enumerate(row):
        label = '—' if np.isnan(value) else f'{value:g}'
        if len(label) > 4:
            label = f'{value:.2f}'
        color = 'white' if value > .6 else '#202020'
        axes[0].text(j, i, label, ha='center', va='center',
                      fontsize=11, color=color)
axes[0].set(xticks=locations,
            xticklabels=[f'x{i + 1}' for i in locations],
            yticks=range(3), yticklabels=list(metrics),
            title='Положительный знак среди выбранных')
pairs = [(p['model__alpha'], p['model__l1_ratio']) for p in settings]
counts = Counter(pairs)
ordered = sorted(counts)
axes[1].bar(range(len(ordered)), [counts[p] for p in ordered])
axes[1].set(xticks=range(len(ordered)),
            xticklabels=[f'{a:g}\n{r:g}' for a, r in ordered],
            xlabel='α (верхняя строка), доля L1 (нижняя)',
            ylabel='Число выборов из 60', title='Подбор повторялся внутри')
savefig(fig, 'e26_signs_tuning')

# На общих входах сравниваются одни и те же адреса прогноза.
pred_sd = {name: [pstdev(column) for column in zip(*values)]
           for name, values in forecasts.items()}
coef_sd = {name: [pstdev(column) for column in zip(*ws)]
           for name, ws in weights.items()}
covariance = {name: sum_variance(ws, 0, 4, 2., .95)
              for name, ws in weights.items()}
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.7))
names = list(weights)
axes[0].bar(names, [r['variance'] ** .5 for r in covariance.values()])
axes[0].set(ylabel='SD общего эффекта 2w1 + 0.95w5',
            title='Сопоставимые исходные единицы')
axes[1].boxplot([pred_sd[name] for name in names], tick_labels=names)
axes[1].set(ylabel='SD прогноза в единицах продаж',
            title='60 общих входов, без проверки по тесту')
savefig(fig, 'e26_group_predictions')

# Точные четыре набора из ручного примера.
toy = [[1., 0., 0.], [0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]
toy_metrics = stability_metrics(toy, {0, 1})
assert toy_metrics['frequency'] == [.5, .5, .25]
assert toy_metrics['group_any'] == 1. and toy_metrics['group_all'] == 0.
assert abs(toy_metrics['mean_jaccard'] - .25) < 1e-12
assert covariance['ElasticNet']['identity_error'] < 1e-12
full_search = search_model().fit(X, y)
chosen = full_search.best_estimator_
U = chosen.named_steps['scale'].transform(X)
params = chosen.named_steps['model']
own = coordinate_descent(U, y, params.alpha, params.l1_ratio,
                         tol=1e-9, max_iter=30000)
assert own['converged']
replication_gap = float(np.max(np.abs(
    chosen.predict(X) - predict(U, own['coef'], own['intercept']))))
assert replication_gap < 1e-6
RESULT = {'seed': 715, 'B': B, 'subset_size': size,
          'subsets_without_replacement': subsets,
          'metrics_epsilon': metrics, 'metrics_exact_zero': exact,
          'weights_original_units': weights, 'selected_settings': settings,
          'forecast_sd': pred_sd, 'forecasts': forecasts,
          'coefficient_sd': coef_sd,
          'group_variance_identity': covariance, 'toy': toy_metrics,
          'full_train_best': full_search.best_params_,
          'own_status': own['status'],
          'own_optimality': own['optimality_error'],
          'own_library_prediction_gap': replication_gap,
          'max_mc_se_bound': .5 / B ** .5}
FIGURES = ['e26_selection', 'e26_signs_tuning', 'e26_group_predictions']
finish_extra('e26', RESULT)
