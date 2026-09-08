#!/usr/bin/env python3
"""Готовое решение Stepik: урок 02, шаг 009.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines = data.splitlines()
    y = list(map(float, lines[0].split()))
    prediction = list(map(float, lines[1].split()))
    total = 0.0
    for yi, pi in zip(y, prediction):
        total += abs(yi - pi)
    return f'{total / len(y):.6f}'

CASES = ['1 4 8\n3 4 6\n', '0\n0\n', '-2 2\n2 -2\n', '1 2 3 4\n1 2 3 8\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
