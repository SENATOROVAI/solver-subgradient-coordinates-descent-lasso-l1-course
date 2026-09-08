from pure_core import (_mean, _number, _penalties, _predict, _total, 
                       _training_data, _weights, residuals)

def _objective_residual(r, w, lambda1, lambda2):
    loss = _mean([value * value for value in r]) / 2
    l1 = _total(lambda1 * abs(value) for value in w)
    l2 = _total((lambda2 / 2 * value) * value for value in w)
    return _total([loss, l1, l2])


def objective(X, y, w, b=0.0, alpha=0.0, l1_ratio=1.0):
    """MSE/2 + alpha*rho*sum|w| + alpha*(1-rho)*sum(w*w)/2."""
    X, y = _training_data(X, y)
    w = _weights(w, len(X[0]))
    b = _number(b, "b")
    lambda1, lambda2 = _penalties(alpha, l1_ratio)
    r = residuals(y, _predict(X, w, b))
    return _objective_residual(r, w, lambda1, lambda2)
