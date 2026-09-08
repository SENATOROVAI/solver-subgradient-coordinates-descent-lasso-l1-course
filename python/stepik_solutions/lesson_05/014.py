#!/usr/bin/env python3
"""Готовое решение Stepik: урок 05, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines = data.splitlines()
    n, m, steps = map(int, lines[0].split())
    eta, decay = map(float, lines[1].split())
    rows = [list(map(float, line.split())) for line in lines[2:2 + n]]
    X = [row[:m] for row in rows]
    y = [row[m] for row in rows]
    weights = [0.0] * m
    b = 0.0
    for t in range(1, steps + 1):
        signs = []
        for row, target in zip(X, y):
            error = sum(x * w for x, w in zip(row, weights)) + b - target
            signs.append((error > 0) - (error < 0))
        gradient = [sum(signs[i] * X[i][j] for i in range(n)) / n for j in range(m)]
        gb = sum(signs) / n
        rate = eta / (t ** decay)
        weights = [w - rate * g for w, g in zip(weights, gradient)]
        b -= rate * gb
    loss = sum(abs(target - (sum(x * w for x, w in zip(row, weights)) + b)) for row, target in zip(X, y)) / n
    return fmt(weights + [b, loss])

CASES = ['3 2 1\n0.3 0\n1 0 3\n0 1 -2\n1 1 1\n', '4 2 5\n0.2 0.5\n1 0 3\n0 1 -2\n1 1 1\n-1 2 -7\n', '5 3 8\n0.15 0.6\n1 0 0 2\n0 1 0 -1\n0 0 1 3\n1 1 1 4\n-1 2 -1 -7\n', '3 2 4\n0.1 0\n0 0 0\n1 0 1\n0 1 1\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
