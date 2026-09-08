from pure_core import (_number)

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
