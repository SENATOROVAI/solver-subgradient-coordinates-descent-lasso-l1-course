"""Явный мост к SciPy: нормировка prior Elastic Net."""
import math
from scipy.special import erfcx


def log_z(a, d):
    if not all(math.isfinite(v) and v >= 0 for v in [a, d]):
        raise ValueError("Нужны конечные a,d >= 0")
    if a == d == 0:
        raise ValueError("При a=d=0 нормируемой плотности нет")
    if d == 0:
        return math.log(2) - math.log(a)
    base = 0.5 * (math.log(2 * math.pi) - math.log(d))
    if a == 0:
        return base
    return base + math.log(float(erfcx(a / math.sqrt(2 * d))))


def prior_pdf(t, a, d):
    return math.exp(-a * abs(t) - d * t * t / 2 - log_z(a, d))
