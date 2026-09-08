from pure_core import (_mean, residuals)

def mse(y, prediction):
    """Средний квадрат ошибки, без множителя 1/2."""
    return _mean([r * r for r in residuals(y, prediction)])
