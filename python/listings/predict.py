from pure_core import (_matrix, _number, _total, _weights)

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
