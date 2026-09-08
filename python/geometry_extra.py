"""Метрики групп и устойчивости: только стандартная библиотека."""
import math
from itertools import combinations
from statistics import mean, pvariance


def grouping_pair(X, y, w, j, k, lambda2, b=0.0):
    """Проверка близости только активной пары одного знака."""
    if lambda2 <= 0:
        raise ValueError("Для границы требуется lambda2 > 0")
    if not w[j] or not w[k] or w[j] * w[k] <= 0:
        return {"valid": False, "reason": "Нужны активные равные знаки"}
    n = len(y)
    r = [yi - b - sum(a * v for a, v in zip(row, w))
         for row, yi in zip(X, y)]
    difference = [row[j] - row[k] for row in X]
    norm_difference = math.sqrt(sum(v * v for v in difference))
    norm_r = math.sqrt(sum(v * v for v in r))
    direct = norm_difference * norm_r / (n * lambda2)
    identity = sum(a * v for a, v in zip(difference, r)) / n
    qj = sum(row[j] ** 2 for row in X) / n
    qk = sum(row[k] ** 2 for row in X) / n
    cross = sum(row[j] * row[k] for row in X) / n
    centered = abs(sum(row[j] for row in X) / n) < 1e-10
    centered = centered and abs(sum(row[k] for row in X) / n) < 1e-10
    normalized = centered and abs(qj - 1) < 1e-10
    normalized = normalized and abs(qk - 1) < 1e-10
    correlation_bound = None
    if normalized:
        correlation_bound = norm_r / math.sqrt(n) / lambda2
        correlation_bound *= math.sqrt(max(0.0, 2 * (1 - cross)))
    return {"valid": True, "difference": abs(w[j] - w[k]),
            "direct_bound": direct,
            "correlation_bound": correlation_bound,
            "identity_error": abs(lambda2 * (w[j] - w[k])
                                  - identity),
            "normalized": normalized}


def jaccard(left, right):
    """Два пустых выбранных набора считаем совпавшими."""
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def stability_metrics(weights, group, threshold=0.0):
    """Частоты относятся к этому протоколу повторного обучения."""
    if not weights or threshold < 0:
        raise ValueError("Нужны повторы и threshold >= 0")
    count, width = len(weights), len(weights[0])
    supports = [{j for j, v in enumerate(row) if abs(v) > threshold}
                for row in weights]
    frequency = [mean(j in active for active in supports)
                 for j in range(width)]
    positive = [mean(row[j] > threshold for row in weights)
                for j in range(width)]
    conditional = [pos / freq if freq else None
                   for pos, freq in zip(positive, frequency)]
    group = set(group)
    pairs = [jaccard(a, b) for a, b in combinations(supports, 2)]
    return {"threshold": threshold, "frequency": frequency,
            "positive_frequency": positive,
            "positive_if_selected": conditional,
            "mc_standard_error": [math.sqrt(f * (1 - f) / count)
                                  for f in frequency],
            "group_any": mean(bool(a & group) for a in supports),
            "group_all": mean(group <= a for a in supports),
            "jaccard_values": pairs,
            "mean_jaccard": mean(pairs) if pairs else None}


def sum_variance(weights, j, k, aj=1.0, ak=1.0):
    """Точная арифметика массива с делителем B."""
    left, right = [w[j] for w in weights], [w[k] for w in weights]
    ml, mr = mean(left), mean(right)
    covariance = mean((u - ml) * (v - mr)
                      for u, v in zip(left, right))
    direct = pvariance([aj * u + ak * v
                        for u, v in zip(left, right)])
    decomposed = aj * aj * pvariance(left) + ak * ak * pvariance(right)
    decomposed += 2 * aj * ak * covariance
    return {"variance": direct, "decomposed": decomposed,
            "covariance": covariance,
            "identity_error": abs(direct - decomposed)}
