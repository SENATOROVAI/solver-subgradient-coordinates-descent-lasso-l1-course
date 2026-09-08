from pure_core import (coordinate_descent)

def lasso_cd(X, y, alpha=0.1, **kwargs):
    return coordinate_descent(X, y, alpha=alpha, l1_ratio=1.0,
                              **kwargs)
