"""Веса, квантили и случайные шаги MAE: стандартная библиотека."""
import math
import random
from pure_core import _training_data, _vector, _number, _integer
from pure_core import _weights, _predict, _mean


def checked_weights(weights, n):
    values = [1.0] * n if weights is None else _vector(weights, "weights")
    if len(values) != n or any(value < 0 for value in values):
        raise ValueError("Веса: нужна длина n и неотрицательные значения")
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Сумма весов должна быть конечной и положительной")
    return values, total


def weighted_mae(y, prediction, weights=None):
    y = _vector(y, "y")
    prediction = _vector(prediction, "prediction")
    if len(y) != len(prediction):
        raise ValueError("Разное число целей и прогнозов")
    values, total = checked_weights(weights, len(y))
    return math.fsum((a / total) * abs(v - p)
                     for a, v, p in zip(values, y, prediction))


def quantile_interval(y, level=0.5, weights=None):
    """Все минимизаторы взвешенной эмпирической pinball loss.

    Нулевые веса удаляются. Точное равенство накопленной массы
    порогу означает интервал; для дробных весов действует float.
    """
    y = _vector(y, "y")
    level = _number(level, "level")
    if not 0 < level < 1:
        raise ValueError("Нужно 0 < level < 1")
    values, total = checked_weights(weights, len(y))
    pairs = sorted((v, a) for v, a in zip(y, values) if a > 0)
    threshold = level * total
    cumulative = 0.0
    for j, (value, weight) in enumerate(pairs):
        cumulative = math.fsum([cumulative, weight])
        if cumulative >= threshold:
            if cumulative == threshold and j + 1 < len(pairs):
                return value, pairs[j + 1][0]
            return value, value
    return pairs[-1][0], pairs[-1][0]


def pinball(y, prediction, level=0.5):
    y = _vector(y, "y")
    prediction = _vector(prediction, "prediction")
    level = _number(level, "level")
    if len(y) != len(prediction) or not 0 < level < 1:
        raise ValueError("Проверьте длины и 0 < level < 1")
    residual = [v - p for v, p in zip(y, prediction)]
    return _mean([max(level * r, (level - 1) * r)
                  for r in residual])


def mae_gradient(X, y, w, b=0.0, weights=None):
    X, y = _training_data(X, y)
    w = _weights(w, len(X[0]))
    b = _number(b, "b")
    values, total = checked_weights(weights, len(y))
    error = [p - v for p, v in zip(_predict(X, w, b), y)]
    signs = [(value > 0) - (value < 0) for value in error]
    factors = [a * s / total for a, s in zip(values, signs)]
    grad = [math.fsum(row[j] * f for row, f in zip(X, factors))
            for j in range(len(w))]
    return grad, math.fsum(factors)


def stochastic_mae(X, y, weights=None, batch_size=1,
                   sampling="uniform", updates=1000, step=0.2,
                   power=0.6, seed=42, evaluate_every=20):
    """Выборка с возвращением; возвращает лучшее из проверенных.

    Полная оценка считается отдельно от строк, использованных
    для обновлений. Статус всегда сообщает о конечном бюджете.
    """
    X, y = _training_data(X, y)
    n, p = len(y), len(X[0])
    values, total = checked_weights(weights, n)
    updates = _integer(updates, "updates", minimum=1)
    batch_size = _integer(batch_size, "batch_size", minimum=1)
    evaluate_every = _integer(evaluate_every, "evaluate_every", minimum=1)
    step, power = _number(step, "step"), _number(power, "power")
    if step <= 0 or power < 0:
        raise ValueError("Нужно step > 0 и power >= 0")
    if sampling not in {"full", "uniform", "weighted"}:
        raise ValueError("sampling: full, uniform или weighted")
    rng = random.Random(_integer(seed, "seed", minimum=None))
    w, b = [0.0] * p, 0.0
    avg, avg_b, mass = [0.0] * p, 0.0, 0.0
    best_w, best_b = w[:], b
    best = weighted_mae(y, _predict(X, w, b), values)
    history = [{"updates": 0, "rows": 0, "loss": best}]
    for k in range(1, updates + 1):
        rate = step * math.exp(-power * math.log(k))
        # Усредняются точки ДО шага, с весами eta_k.
        mass += rate
        avg = [a + rate * v for a, v in zip(avg, w)]
        avg_b += rate * b
        if sampling == "full":
            ids = range(n)
            factors = [a / total for a in values]
        else:
            if sampling == "uniform":
                ids = [rng.randrange(n) for _ in range(batch_size)]
                factors = [n * values[i] / (total * batch_size)
                           for i in ids]
            else:
                ids = rng.choices(range(n), weights=values,
                                  k=batch_size)
                factors = [1.0 / batch_size] * batch_size
        grad, grad_b = [0.0] * p, 0.0
        for i, factor in zip(ids, factors):
            error = math.fsum([b] + [a * v for a, v in zip(X[i], w)])
            error -= y[i]
            sign = (error > 0) - (error < 0)
            grad = [g + factor * sign * x
                    for g, x in zip(grad, X[i])]
            grad_b += factor * sign
        w = [v - rate * g for v, g in zip(w, grad)]
        b -= rate * grad_b
        if k % evaluate_every == 0 or k == updates:
            loss = weighted_mae(y, _predict(X, w, b), values)
            history.append({"updates": k, "rows": k * len(ids),
                            "loss": loss})
            if loss < best:
                best_w, best_b, best = w[:], b, loss
    average_w, average_b = [a / mass for a in avg], avg_b / mass
    return {"coef": best_w, "intercept": best_b, "best_loss": best,
            "last_coef": w, "last_intercept": b,
            "last_loss": history[-1]["loss"], "history": history,
            "average_coef": average_w, "average_intercept": average_b,
            "average_loss": weighted_mae(
                y, _predict(X, average_w, average_b), values),
            "update_rows": history[-1]["rows"],
            "diagnostic_rows": len(history) * n,
            "n_iter": updates, "converged": False,
            "status": "budget_exhausted"}
