from pure_core import (_dot)

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
