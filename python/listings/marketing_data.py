import math
import random
from pure_core import (_dot, _integer, _number)

def marketing_data(n=160, p=8, seed=42, noise=1.0, correlation=0.95):
    """Синтетическая связь признаков с продажами, не причинный эффект."""
    n = _integer(n, "n", minimum=1)
    p = _integer(p, "p", minimum=3)
    seed = _integer(seed, "seed", minimum=None)
    noise = _number(noise, "noise")
    correlation = _number(correlation, "correlation")
    if noise < 0 or not -1 <= correlation <= 1:
        raise ValueError("Нужно noise >= 0 и -1 <= correlation <= 1")
    rng = random.Random(seed)
    beta = [3.0, -2.0, 1.0] + [0.0] * (p - 3)
    X, y = [], []
    for _ in range(n):
        z = [rng.gauss(0, 1) for _ in range(p)]
        row = z[:]
        row[0] = 10 + 2 * z[0]
        row[1] = 5 + z[1]
        row[2] = 3 + 1.5 * z[2]
        if p >= 5:
            fresh = rng.gauss(0, 1)
            row[4] = correlation * z[0]
            row[4] += math.sqrt(1 - correlation * correlation) * fresh
        target = 20 + _dot(row, beta) + noise * rng.gauss(0, 1)
        X.append(row)
        y.append(_number(target, "y"))
    return X, y, beta
