import math
from pure_core import (_integer, _mean, _number, _predict, _total, 
                       _training_data, _weights, mae, median)

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
