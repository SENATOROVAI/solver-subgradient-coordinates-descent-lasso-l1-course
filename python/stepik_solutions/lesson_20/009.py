#!/usr/bin/env python3
"""Готовое решение Stepik: урок 20, шаг 009.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    rows=[s.split() for s in data.splitlines()]; return min(rows,key=lambda r:float(r[1]))[0]

CASES = ['Mean 8\nOLS 1.2\nLasso 0.95\nRidge 1.0\nElasticNet 0.98\n', 'A 1\nB 2\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
