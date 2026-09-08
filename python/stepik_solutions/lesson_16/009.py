#!/usr/bin/env python3
"""Готовое решение Stepik: урок 16, шаг 009.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    alpha,rho=map(float,data.split()); return f'{alpha*rho:.6f} {alpha*(1-rho):.6f}'

CASES = ['0.5 0.2\n', '0.5 0.8\n', '2.2 0.09090909090909091\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
