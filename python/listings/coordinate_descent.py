from pure_core import (_balance_error, _dot, _integer, _mean, 
                       _number, _objective_residual, _penalties, 
                       _predict, _training_data, _weights, residuals, 
                       soft_threshold)

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
