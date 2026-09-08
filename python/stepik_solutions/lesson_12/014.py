#!/usr/bin/env python3
"""Готовое решение Stepik: урок 12, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines = data.splitlines()
    n, m, cycles = map(int, lines[0].split())
    rows = [list(map(float, line.split())) for line in lines[1:1 + n]]
    X = [row[:m] for row in rows]
    y = [row[m] for row in rows]
    weights = [0.0] * m
    prediction = [0.0] * n
    for _ in range(cycles):
        for j in range(m):
            denominator = sum(X[i][j] ** 2 for i in range(n))
            if denominator == 0:
                new_weight = 0.0
            else:
                numerator = sum(X[i][j] * (y[i] - prediction[i] + X[i][j] * weights[j]) for i in range(n))
                new_weight = numerator / denominator
            difference = new_weight - weights[j]
            for i in range(n):
                prediction[i] += X[i][j] * difference
            weights[j] = new_weight
    return fmt(weights)

CASES = ['2 2 1\n1 0 2\n0 1 -3\n', '3 2 5\n1 1 3\n2 0 4\n0 2 2\n', '3 2 3\n1 0 2\n2 0 4\n3 0 6\n', '4 3 8\n1 0 1 2\n0 1 1 1\n1 1 0 3\n-1 1 1 -1\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
