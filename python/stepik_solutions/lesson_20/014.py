#!/usr/bin/env python3
"""Готовое решение Stepik: урок 20, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines = data.splitlines()
    first = lines[0].split()
    n, m = int(first[0]), int(first[1])
    sigma2, tau2 = float(first[2]), float(first[3])
    prior_mean = list(map(float, lines[1].split()))
    rows = [list(map(float, line.split())) for line in lines[2:2 + n]]
    X = [row[:m] for row in rows]
    y = [row[m] for row in rows]
    k = int(lines[2 + n])
    new_X = [list(map(float, line.split())) for line in lines[3 + n:3 + n + k]]
    precision = [[sum(X[r][i] * X[r][j] for r in range(n)) / sigma2 + ((1.0 / tau2) if i == j else 0.0) for j in range(m)] for i in range(m)]
    right = [sum(X[r][i] * y[r] for r in range(n)) / sigma2 + prior_mean[i] / tau2 for i in range(m)]
    augmented = [precision[i] + [1.0 if i == j else 0.0 for j in range(m)] + [right[i]] for i in range(m)]
    for column in range(m):
        pivot = max(range(column, m), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(m):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[column])]
    covariance = [row[m:2 * m] for row in augmented]
    posterior_mean = [row[-1] for row in augmented]
    variances = [covariance[i][i] for i in range(m)]
    predictions = [sum(x * w for x, w in zip(row, posterior_mean)) for row in new_X]
    return fmt(posterior_mean + variances + predictions)

CASES = ['3 2 1 1\n0 0\n1 0 2\n0 1 -1\n1 1 1\n2\n1 0\n1 1\n', '4 2 2 0.5\n1 -1\n1 1 3\n2 0 4\n0 2 -2\n-1 1 -3\n2\n2 1\n-1 -1\n', '4 3 1.5 2\n0 0 0\n1 0 0 1\n0 1 0 -1\n0 0 1 2\n1 1 1 2\n2\n1 1 0\n0 1 1\n', '3 2 4 10\n2 -2\n1 1 1\n2 2 2\n-1 -1 -1\n1\n3 -1\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
