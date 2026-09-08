from pure_core import (coordinate_descent)

def elasticnet_cd(X, y, alpha=0.1, l1_ratio=0.5, **kwargs):
    return coordinate_descent(X, y, alpha=alpha, l1_ratio=l1_ratio,
                              **kwargs)
