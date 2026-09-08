#!/usr/bin/env python3
"""Готовое решение Stepik: урок 06, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines = data.splitlines()
    n, m = map(int, lines[0].split())
    rows = [list(map(float, line.split())) for line in lines[1:1 + n]]
    X = [row[:m] for row in rows]
    y = [row[m] for row in rows]
    matrix = [[sum(X[i][j] * X[i][k] for i in range(n)) for k in range(m)] for j in range(m)]
    right = [sum(X[i][j] * y[i] for i in range(n)) for j in range(m)]
    augmented = [matrix[i] + [right[i]] for i in range(m)]
    for column in range(m):
        pivot = max(range(column, m), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(m):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[column])]
    weights = [augmented[i][-1] for i in range(m)]
    residuals = [y[i] - sum(X[i][j] * weights[j] for j in range(m)) for i in range(n)]
    variance = sum(value * value for value in residuals) / n
    return fmt(weights + [variance])

CASES = ['4 2\n1 0 2\n0 1 -3\n1 1 -0.5\n2 -1 7\n', '5 2\n1 1 2\n2 0 4.5\n0 2 -2\n-1 1 -3\n3 -1 8\n', '5 3\n1 0 0 1\n0 1 0 2\n0 0 1 -1\n1 1 1 2.5\n2 -1 1 -0.5\n', '6 2\n1 -1 3\n1 0 1.2\n1 1 -0.8\n1 2 -3.1\n1 3 -4.9\n1 4 -7.2\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
