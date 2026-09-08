"""Одинаковые проходы CD: пересчёт прогноза и сохранение остатка.

Считаем умножения внутри координатных обновлений. Центрирование,
нормы столбцов и итоговая проверка исключены у обоих вариантов.
Это счёт операций, а не измерение времени выполнения программы.
"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import (
    _dot, _integer, _mean, _penalties, _total, _training_data,
    coordinate_descent, marketing_data, objective,
    scale_fit, scale_transform, soft_threshold,
)


def fixed_cd(X, y, alpha=.2, l1_ratio=.5, epochs=4, cached=True):
    """Точно epochs полных проходов, старт w=0; b профилируется."""
    X, y = _training_data(X, y)
    epochs = _integer(epochs, "epochs")
    if not isinstance(cached, bool):
        raise ValueError("cached должен быть bool")
    l1, l2 = _penalties(alpha, l1_ratio)
    n, p = len(y), len(X[0])
    means = [_mean(list(c)) for c in zip(*X)]
    ym = _mean(y)
    A = [[v - m for v, m in zip(row, means)] for row in X]
    yc = [v - ym for v in y]
    q = [_total(row[j] ** 2 for row in A) / n for j in range(p)]
    w, r = [0.] * p, yc[:]
    counts = dict(prediction=0, correlation=0, residual_update=0)
    for _ in range(epochs):
        for j in range(p):
            if cached:
                partial = [r[i] + A[i][j] * w[j] for i in range(n)]
                counts["residual_update"] += n
            else:
                partial = [yc[i] - _total(A[i][k] * w[k]
                           for k in range(p) if k != j) for i in range(n)]
                counts["prediction"] += n * (p - 1)
            z = _total(A[i][j] * partial[i] for i in range(n)) / n
            counts["correlation"] += n
            w[j] = soft_threshold(z, l1) / (q[j] + l2) if q[j] else 0.
            if cached:
                r = [partial[i] - A[i][j] * w[j] for i in range(n)]
                counts["residual_update"] += n
    b = ym - _dot(means, w)
    return dict(coef=w, intercept=b, epochs=epochs, counters=counts,
                products=sum(counts.values()),
                objective=objective(X, y, w, b, alpha, l1_ratio))


def experiment():
    rows = []
    for p in [4, 8, 16, 32]:
        X, y, _ = marketing_data(n=48, p=p, seed=1200 + p)
        X = scale_transform(X, scale_fit(X))
        naive = fixed_cd(X, y, cached=False)
        saved = fixed_cd(X, y, cached=True)
        reference = coordinate_descent(X, y, alpha=.2, l1_ratio=.5,
                                       max_iter=4, tol=0.)
        delta = max(abs(a - b) for a, b in
                    zip(naive["coef"], saved["coef"]))
        delta_ref = max(abs(a - b) for a, b in
                        zip(saved["coef"], reference["coef"]))
        assert reference["n_iter"] == 4
        assert delta < 1e-12 and delta_ref < 1e-12
        assert naive["products"] == 4 * 48 * p * p
        assert saved["products"] == 4 * 48 * p * 3
        rows.append(dict(n=48, p=p, epochs=4,
                         naive_products=naive["products"],
                         cached_products=saved["products"],
                         product_ratio=p / 3,
                         max_coef_difference=delta,
                         max_reference_difference=delta_ref,
                         naive_objective=naive["objective"],
                         cached_objective=saved["objective"]))
    return {"rows": rows, "count_scope": "coordinate multiplications"}


if __name__ == "__main__":
    RESULT = experiment()
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
