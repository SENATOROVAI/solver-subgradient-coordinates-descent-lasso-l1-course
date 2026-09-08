#!/usr/bin/env python3
"""Готовое решение Stepik: урок 18, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    rows=[list(map(float,s.split())) for s in data.splitlines()]; best=min(rows,key=lambda r:r[2]); return f'{best[0]:.6f} {best[1]:.6f}'

CASES = ['0.03 0.2 1.2\n0.03 1 1.1\n0.15 0.6 1.4\n', '0.1 0.5 2\n1 1 3\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
