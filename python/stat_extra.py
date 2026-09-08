"""Статистические формулы: только стандартная библиотека Python."""
import math
from statistics import NormalDist, mean, median


def normal_pdf(x, mu=0.0, sigma=1.0):
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Нужно sigma > 0, конечное")
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (
        math.sqrt(2 * math.pi) * sigma)


def normal_cdf(x, mu=0.0, sigma=1.0):
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Нужно sigma > 0, конечное")
    return NormalDist(mu, sigma).cdf(x)


def laplace_pdf(x, mu=0.0, scale=1.0):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Нужно scale > 0, конечное")
    return math.exp(-abs(x - mu) / scale) / (2 * scale)


def laplace_cdf(x, mu=0.0, scale=1.0):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Нужно scale > 0, конечное")
    z = (x - mu) / scale
    return 0.5 * math.exp(z) if z < 0 else 1 - 0.5 * math.exp(-z)


def laplace_ppf(q, mu=0.0, scale=1.0):
    if not 0 < q < 1 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Нужны 0 < q < 1 и scale > 0, конечное")
    z = math.log(2 * q) if q < 0.5 else -math.log(2 * (1 - q))
    return mu + scale * z


def location_scale_mle(values):
    """Делитель n; при нулевом масштабе положительной MLE нет."""
    values = list(values)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Нужна непустая конечная выборка")
    mu_n, mu_l = mean(values), median(values)
    variance = mean([(v - mu_n) ** 2 for v in values])
    scale = mean([abs(v - mu_l) for v in values])
    if variance == 0 or scale == 0:
        raise ValueError("Нет конечной положительной MLE масштаба")
    sigma = math.sqrt(variance)
    nll_n = sum(math.log(sigma) + 0.5 * math.log(2 * math.pi)
                + 0.5 * ((v - mu_n) / sigma) ** 2 for v in values)
    nll_l = sum(math.log(2 * scale) + abs(v - mu_l) / scale
                for v in values)
    return {"normal_mu": mu_n, "normal_sigma": sigma,
            "normal_variance": variance, "normal_nll": nll_n,
            "laplace_mu": mu_l, "laplace_scale": scale,
            "laplace_nll": nll_l}


def shifted_laplace_map(center, curvature, penalty, prior_mu=0.0):
    """Минимум q*(w-c)^2/2 + penalty*|w-m|; q > 0."""
    args = [center, curvature, penalty, prior_mu]
    if not all(math.isfinite(v) for v in args):
        raise ValueError("Аргументы должны быть конечными")
    if curvature <= 0 or penalty < 0:
        raise ValueError("Нужны curvature > 0, penalty >= 0")
    offset = center - prior_mu
    shrink = max(abs(offset) - penalty / curvature, 0.0)
    return prior_mu + math.copysign(shrink, offset)


def trapezoid(x, y):
    """Квадратура на общей возрастающей сетке."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Нужны две одинаковые длины >= 2")
    if any(b <= a for a, b in zip(x, x[1:])):
        raise ValueError("Сетка должна возрастать")
    return math.fsum((b - a) * (u + v) / 2
                     for a, b, u, v in zip(x, x[1:], y, y[1:]))
