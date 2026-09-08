"""Два разбиения Elastic Net: один и тот же следующий шаг."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import soft_threshold


def two_steps(w, correlations, L0, lambda1, lambda2):
    """Фрагмент обновления: correlations=Xc^T(yc-Xc*w)/n, L0>0."""
    smooth_step = 1 / (L0 + lambda2)
    prox_step = 1 / L0
    smooth = [soft_threshold(v + smooth_step * (c - lambda2 * v),
                             smooth_step * lambda1)
              for v, c in zip(w, correlations)]
    prox = [soft_threshold(v + prox_step * c, prox_step * lambda1)
            / (1 + prox_step * lambda2)
            for v, c in zip(w, correlations)]
    return smooth, prox


a, b = two_steps([.4, -.2], [1.5, -.7], 3., .3, .2)
assert max(abs(x - y) for x, y in zip(a, b)) < 1e-15
print({"smooth_l2": a, "prox_all": b})
