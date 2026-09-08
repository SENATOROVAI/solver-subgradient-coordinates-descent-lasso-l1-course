from pure_core import (_matrix, _number, _scale_stats)

def scale_transform(X, stats):
    X = _matrix(X)
    mean, scale = _scale_stats(stats, len(X[0]))
    return [[_number((x - m) / s, "Масштабированный признак")
             for x, m, s in zip(row, mean, scale)] for row in X]
