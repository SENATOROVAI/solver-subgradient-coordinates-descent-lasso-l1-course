"""Взвешенный CD через строки: только стандартная библиотека."""
import math
from pure_core import (_training_data, _vector, _number,
                       coordinate_descent, predict)


def prepare_weighted(X, y, sample_weight, fit_intercept=True):
    """Центрировать по весам, затем умножить строки на sqrt(q)."""
    X, y = _training_data(X, y)
    s = _vector(sample_weight, "sample_weight")
    if len(s) != len(y) or any(v < 0 for v in s):
        raise ValueError("Нужны n неотрицательных весов")
    total = math.fsum(s)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("Сумма весов должна быть конечной и > 0")
    if not isinstance(fit_intercept, bool):
        raise ValueError("fit_intercept должен быть bool")
    n, p = len(y), len(X[0])
    q = [n * (v / total) for v in s]
    means = [0.] * p
    ym = 0.
    if fit_intercept:
        means = [math.fsum(v * row[j] / total
                          for v, row in zip(s, X))
                 for j in range(p)]
        ym = math.fsum(v * t / total for v, t in zip(s, y))
    Z = [[_number(math.sqrt(v) * (x - m), "X weighted")
          for x, m in zip(row, means)] for row, v in zip(X, q)]
    t = [_number(math.sqrt(v) * (a - ym), "y weighted")
         for a, v in zip(y, q)]
    return {"X": Z, "y": t, "q": q,
            "X_mean": means, "y_mean": ym}


def weighted_cd(X, y, sample_weight, alpha=.1, l1_ratio=.5,
                fit_intercept=True, max_iter=10000, tol=1e-9):
    prepared = prepare_weighted(X, y, sample_weight, fit_intercept)
    fit = coordinate_descent(
        prepared["X"], prepared["y"], alpha=alpha,
        l1_ratio=l1_ratio, fit_intercept=False,
        max_iter=max_iter, tol=tol)
    b = prepared["y_mean"] - math.fsum(
        m * w for m, w in zip(prepared["X_mean"], fit["coef"]))
    fit["intercept"] = b
    fit["weight_sum_normalized"] = math.fsum(prepared["q"])
    return fit


def weighted_objective(X, y, s, w, b, alpha=.1, l1_ratio=.5):
    """Внешняя проверка цели в исходных координатах."""
    prepare_weighted(X, y, s, fit_intercept=False)
    r = [a - v for a, v in zip(y, predict(X, w, b))]
    loss = math.fsum(v * e * e for v, e in zip(s, r))
    loss /= 2 * math.fsum(s)
    penalty = alpha * l1_ratio * math.fsum(abs(a) for a in w)
    penalty += alpha * (1 - l1_ratio) / 2 * math.fsum(
        a * a for a in w)
    return loss + penalty
