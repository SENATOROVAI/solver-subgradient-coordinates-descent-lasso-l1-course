"""Учебные алгоритмы: только стандартная библиотека Python."""

import math
import numbers
import random


def _number(value, name):
    """Конечное вещественное число; bool не считается числом."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name}: нужно вещественное число")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        message = f"{name}: число вне допустимого диапазона"
        raise ValueError(message) from exc
    if not math.isfinite(result):
        raise ValueError(f"{name}: число должно быть конечным")
    return result


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name}: нужно целое число")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}: минимум {minimum}")
    return int(value)


def _vector(values, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name}: нужен непустой список чисел")
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(f"{name}: нужен список чисел") from exc
    if not values:
        raise ValueError(f"{name}: пустой список")
    return [_number(value, name) for value in values]


def _matrix(X):
    if isinstance(X, (str, bytes)):
        raise ValueError("X: нужна непустая прямоугольная матрица")
    try:
        rows = list(X)
    except TypeError as exc:
        raise ValueError("X: нужна матрица") from exc
    if not rows:
        raise ValueError("X: пустая матрица")
    rows = [_vector(row, "X") for row in rows]
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("X: строки разной длины")
    return rows


def _training_data(X, y):
    X = _matrix(X)
    y = _vector(y, "y")
    if len(X) != len(y):
        raise ValueError("X и y: разное число наблюдений")
    return X, y


def _weights(w, p):
    w = _vector(w, "w")
    if len(w) != p:
        raise ValueError("w: длина не совпадает с числом признаков")
    return w


def _total(values):
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Переполнение при вычислении суммы") from exc
    return _number(result, "Результат вычисления")


def _mean(values):
    if all(value == values[0] for value in values):
        return _number(values[0], "Среднее")
    return _total(value / len(values) for value in values)


def _dot(a, b):
    return _total(x * y for x, y in zip(a, b))


def _predict(X, w, b):
    return [_total([b, _dot(row, w)]) for row in X]


def predict(X, w, b=0.0):
    """Прогноз для каждой строки: скалярное произведение плюс b."""
    X = _matrix(X)
    w = _weights(w, len(X[0]))
    b = _number(b, "b")
    return _predict(X, w, b)


def residuals(y, prediction):
    """Остаток имеет знак: наблюдение минус прогноз."""
    y = _vector(y, "y")
    prediction = _vector(prediction, "prediction")
    if len(y) != len(prediction):
        raise ValueError("y и prediction: разная длина")
    return [_number(a - b, "Остаток") for a, b in zip(y, prediction)]


def mae(y, prediction):
    """Средняя абсолютная ошибка."""
    return _mean([abs(r) for r in residuals(y, prediction)])


def mse(y, prediction):
    """Средний квадрат ошибки, без множителя 1/2."""
    return _mean([r * r for r in residuals(y, prediction)])


def median(values):
    values = sorted(_vector(values, "values"))
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return values[middle - 1] / 2 + values[middle] / 2


def grid_mae(X, y, w_values, b_values):
    """Полный перебор пар (w, b) для одного признака."""
    X, y = _training_data(X, y)
    if len(X[0]) != 1:
        raise ValueError("grid_mae: нужен ровно один признак")
    w_values = _vector(w_values, "w_values")
    b_values = _vector(b_values, "b_values")
    records = []
    for w in w_values:
        for b in b_values:
            loss = mae(y, _predict(X, [w], b))
            records.append((w, b, loss))
    w, b, loss = min(records, key=lambda record: record[2])
    return {"coef": [w], "intercept": b, "loss": loss,
            "records": records}


def fit_mae(X, y, steps=1500, step=0.1, decay=0.6,
            w0=None, b0=None):
    """Шаги по знакам ошибок; сохраняем лучший посещённый вариант.

    История показывает фактический путь. Исчерпание бюджета шагов
    не служит доказательством нахождения минимума.
    """
    X, y = _training_data(X, y)
    steps = _integer(steps, "steps")
    step = _number(step, "step")
    decay = _number(decay, "decay")
    if step <= 0 or decay < 0:
        raise ValueError("Нужно step > 0 и decay >= 0")
    n, p = len(y), len(X[0])
    w = [0.0] * p if w0 is None else _weights(w0, p)
    b = median(y) if b0 is None else _number(b0, "b0")
    loss = mae(y, _predict(X, w, b))
    history = [loss]
    parameter_history = [w[:] + [b]]
    best_w, best_b, best_loss = w[:], b, loss
    for iteration in range(1, steps + 1):
        prediction = _predict(X, w, b)
        signs = [(v > t) - (v < t) for v, t in zip(prediction, y)]
        gradients = [
            _total(X[i][j] * signs[i] / n for i in range(n))
            for j in range(p)
        ]
        # exp(-decay * log(iteration)) avoids power overflow.
        rate = step * math.exp(-decay * math.log(iteration))
        w = [_number(v - rate * g, "w") for v, g in zip(w, gradients)]
        b = _number(b - rate * _mean(signs), "b")
        loss = mae(y, _predict(X, w, b))
        history.append(loss)
        parameter_history.append(w[:] + [b])
        if loss < best_loss:
            best_w, best_b, best_loss = w[:], b, loss
    return {
        "coef": best_w, "intercept": best_b, "best_loss": best_loss,
        "history": history, "parameter_history": parameter_history,
        "n_iter": steps, "status": "max_steps", "converged": False,
    }


def scale_fit(X):
    """Средние и стандартные отклонения с делителем n."""
    X = _matrix(X)
    means = [_mean(list(column)) for column in zip(*X)]
    scales = []
    for column, mean in zip(zip(*X), means):
        centered = [value - mean for value in column]
        variance = _mean([value * value for value in centered])
        scale = math.sqrt(variance)
        scales.append(scale if scale > 0 else 1.0)
    return {"mean": means, "scale": scales}


def _scale_stats(stats, p):
    if not isinstance(stats, dict):
        raise ValueError("stats: нужен словарь mean и scale")
    if "mean" not in stats or "scale" not in stats:
        raise ValueError("stats: отсутствует mean или scale")
    mean = _weights(stats["mean"], p)
    scale = _weights(stats["scale"], p)
    if any(value <= 0 for value in scale):
        raise ValueError("scale: все значения должны быть положительными")
    return mean, scale


def scale_transform(X, stats):
    X = _matrix(X)
    mean, scale = _scale_stats(stats, len(X[0]))
    return [[_number((x - m) / s, "Масштабированный признак")
             for x, m, s in zip(row, mean, scale)] for row in X]


def unscale_coefficients(w, b, stats):
    w = _vector(w, "w")
    b = _number(b, "b")
    mean, scale = _scale_stats(stats, len(w))
    original_w = [_number(v / s, "w") for v, s in zip(w, scale)]
    original_b = _number(b - _dot(mean, original_w), "b")
    return original_w, original_b


def _penalties(alpha, l1_ratio):
    alpha = _number(alpha, "alpha")
    l1_ratio = _number(l1_ratio, "l1_ratio")
    if alpha < 0 or not 0 <= l1_ratio <= 1:
        raise ValueError("Нужно alpha >= 0 и 0 <= l1_ratio <= 1")
    return alpha * l1_ratio, alpha * (1 - l1_ratio)


def _objective_residual(r, w, lambda1, lambda2):
    loss = _mean([value * value for value in r]) / 2
    l1 = _total(lambda1 * abs(value) for value in w)
    l2 = _total((lambda2 / 2 * value) * value for value in w)
    return _total([loss, l1, l2])


def objective(X, y, w, b=0.0, alpha=0.0, l1_ratio=1.0):
    """MSE/2 + alpha*rho*sum|w| + alpha*(1-rho)*sum(w*w)/2."""
    X, y = _training_data(X, y)
    w = _weights(w, len(X[0]))
    b = _number(b, "b")
    lambda1, lambda2 = _penalties(alpha, l1_ratio)
    r = residuals(y, _predict(X, w, b))
    return _objective_residual(r, w, lambda1, lambda2)


def soft_threshold(z, threshold):
    """Уменьшить модуль z на threshold, но не пересечь ноль."""
    z = _number(z, "z")
    threshold = _number(threshold, "threshold")
    if threshold < 0:
        raise ValueError("threshold должен быть неотрицательным")
    if z > threshold:
        return z - threshold
    if z < -threshold:
        return z + threshold
    return 0.0


def _balance_error(columns, r, w, lambda1, lambda2):
    """Абсолютная погрешность баланса для каждой координаты."""
    errors = []
    for column, value in zip(columns, w):
        correlation = _dot(column, r) / len(r)
        smooth = correlation - lambda2 * value
        if value > 0:
            error = abs(smooth - lambda1)
        elif value < 0:
            error = abs(smooth + lambda1)
        else:
            error = max(abs(smooth) - lambda1, 0.0)
        errors.append(error)
    return max(errors)


def coordinate_descent(X, y, alpha=0.1, l1_ratio=1.0,
                       fit_intercept=True, max_iter=5000, tol=1e-8,
                       w0=None, keep_steps=False):
    """Циклические точные обновления отдельных коэффициентов.

    tol ограничивает абсолютную ошибку баланса координат. Это иная
    проверка остановки, чем внутренний критерий библиотечного solver.
    """
    X, y = _training_data(X, y)
    lambda1, lambda2 = _penalties(alpha, l1_ratio)
    max_iter = _integer(max_iter, "max_iter", minimum=1)
    tol = _number(tol, "tol")
    if tol < 0:
        raise ValueError("tol должен быть неотрицательным")
    if not isinstance(fit_intercept, bool):
        raise ValueError("fit_intercept должен быть bool")
    if not isinstance(keep_steps, bool):
        raise ValueError("keep_steps должен быть bool")
    n, p = len(y), len(X[0])
    means = [_mean(list(c)) for c in zip(*X)]
    y_mean = _mean(y)
    if not fit_intercept:
        means, y_mean = [0.0] * p, 0.0
    columns = [[row[j] - means[j] for row in X] for j in range(p)]
    centered_y = [value - y_mean for value in y]
    squares = [_mean([v * v for v in c]) for c in columns]
    w = [0.0] * p if w0 is None else _weights(w0, p)
    w = [value if q > 0 else 0.0 for value, q in zip(w, squares)]
    centered_X = list(zip(*columns))
    r = residuals(centered_y, _predict(centered_X, w, 0.0))
    current = _objective_residual(r, w, lambda1, lambda2)
    history, coef_history = [current], [w[:]]
    trace = []
    if keep_steps:
        trace.append({"iteration": 0, "coordinate": None,
                      "coef": w[:], "objective": current})
    error = _balance_error(columns, r, w, lambda1, lambda2)
    n_iter = 0
    while n_iter < max_iter and error > tol:
        n_iter += 1
        for j, column in enumerate(columns):
            old = w[j]
            # Частичный остаток временно возвращает вклад признака j.
            for i in range(n):
                r[i] += column[i] * old
            z = _dot(column, r) / n
            denominator = _number(squares[j] + lambda2, "Знаменатель")
            if squares[j] == 0:
                w[j] = 0.0
            else:
                w[j] = _number(soft_threshold(z, lambda1) / denominator,
                               "Коэффициент")
            for i in range(n):
                r[i] -= column[i] * w[j]
            if keep_steps:
                value = _objective_residual(r, w, lambda1, lambda2)
                trace.append({"iteration": n_iter, "coordinate": j,
                              "coef": w[:], "objective": value})
        # Пересчёт устраняет накопление округления в остатках.
        r = residuals(centered_y, _predict(centered_X, w, 0.0))
        current = _objective_residual(r, w, lambda1, lambda2)
        history.append(current)
        coef_history.append(w[:])
        error = _balance_error(columns, r, w, lambda1, lambda2)
    converged = error <= tol
    intercept = _number(y_mean - _dot(means, w), "intercept")
    return {
        "coef": w[:], "intercept": intercept, "objective": current,
        "n_iter": n_iter, "converged": converged,
        "status": "converged" if converged else "max_iter",
        "history": history, "coef_history": coef_history,
        "trace": trace, "optimality_error": error,
    }


def lasso_cd(X, y, alpha=0.1, **kwargs):
    return coordinate_descent(X, y, alpha=alpha, l1_ratio=1.0,
                              **kwargs)


def elasticnet_cd(X, y, alpha=0.1, l1_ratio=0.5, **kwargs):
    return coordinate_descent(X, y, alpha=alpha, l1_ratio=l1_ratio,
                              **kwargs)


def marketing_data(n=160, p=8, seed=42, noise=1.0, correlation=0.95):
    """Синтетическая связь признаков с продажами, не причинный эффект."""
    n = _integer(n, "n", minimum=1)
    p = _integer(p, "p", minimum=3)
    seed = _integer(seed, "seed", minimum=None)
    noise = _number(noise, "noise")
    correlation = _number(correlation, "correlation")
    if noise < 0 or not -1 <= correlation <= 1:
        raise ValueError("Нужно noise >= 0 и -1 <= correlation <= 1")
    rng = random.Random(seed)
    beta = [3.0, -2.0, 1.0] + [0.0] * (p - 3)
    X, y = [], []
    for _ in range(n):
        z = [rng.gauss(0, 1) for _ in range(p)]
        row = z[:]
        row[0] = 10 + 2 * z[0]
        row[1] = 5 + z[1]
        row[2] = 3 + 1.5 * z[2]
        if p >= 5:
            fresh = rng.gauss(0, 1)
            row[4] = correlation * z[0]
            row[4] += math.sqrt(1 - correlation * correlation) * fresh
        target = 20 + _dot(row, beta) + noise * rng.gauss(0, 1)
        X.append(row)
        y.append(_number(target, "y"))
    return X, y, beta
