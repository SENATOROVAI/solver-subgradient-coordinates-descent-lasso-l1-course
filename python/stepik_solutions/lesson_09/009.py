#!/usr/bin/env python3
"""Готовое решение Stepik: урок 09, шаг 009.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    from itertools import product
    from math import exp
    lines = data.splitlines()
    first = lines[0].split()
    n, m = int(first[0]), int(first[1])
    sigma2, tau, low, high, step = map(float, first[2:])
    rows = [list(map(float, line.split())) for line in lines[1:1 + n]]
    X = [row[:m] for row in rows]
    y = [row[m] for row in rows]
    k = int(lines[1 + n])
    new_X = [list(map(float, line.split())) for line in lines[2 + n:2 + n + k]]
    axis = [low + i * step for i in range(int(round((high - low) / step)) + 1)]
    candidates = list(product(axis, repeat=m))
    log_weights = []
    for weights in candidates:
        squared_error = 0.0
        for row, target in zip(X, y):
            error = target - sum(x * w for x, w in zip(row, weights))
            squared_error += error * error
        log_weights.append(-squared_error / (2 * sigma2) - sum(abs(w) for w in weights) / tau)
    maximum = max(log_weights)
    posterior_weights = [exp(value - maximum) for value in log_weights]
    normalizer = sum(posterior_weights)
    mean = [sum(weight * candidate[j] for weight, candidate in zip(posterior_weights, candidates)) / normalizer for j in range(m)]
    predictions = [sum(x * w for x, w in zip(row, mean)) for row in new_X]
    return fmt(mean + predictions)

CASES = ['3 2 1 1 -2 2 1\n1 0 2\n0 1 -1\n1 1 1\n2\n1 0\n1 1\n', '4 2 2 0.7 -3 3 1\n1 1 3\n2 0 4\n0 2 -2\n-1 1 -3\n2\n2 1\n-1 -1\n', '4 3 1.5 1.2 -1 1 1\n1 0 0 1\n0 1 0 -1\n0 0 1 1\n1 1 1 1\n2\n1 1 0\n0 1 1\n', '3 2 0.5 2 -2 2 0.5\n1 -1 2\n2 1 1\n-1 2 -2\n1\n3 -1\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
