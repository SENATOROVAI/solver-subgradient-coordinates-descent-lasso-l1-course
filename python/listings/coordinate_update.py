from pure_core import _dot, _number, soft_threshold
from pure_core import _objective_residual

# Фрагмент внутри coordinate_descent; переменные созданы выше.
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
