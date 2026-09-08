from pure_core import (_mean, residuals)

def mae(y, prediction):
    """Средняя абсолютная ошибка."""
    return _mean([abs(r) for r in residuals(y, prediction)])
