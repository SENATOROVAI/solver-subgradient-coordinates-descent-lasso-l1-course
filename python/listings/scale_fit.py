import math
from pure_core import (_matrix, _mean)

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
