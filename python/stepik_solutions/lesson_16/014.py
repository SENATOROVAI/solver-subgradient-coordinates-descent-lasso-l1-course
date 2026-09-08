#!/usr/bin/env python3
"""Готовое решение Stepik: урок 16, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    l1,l2=map(float,data.split()); alpha=l1+l2; rho=l1/alpha if alpha else 0; return f'{alpha:.6f} {rho:.6f}'

CASES = ['0.2 2\n', '0.15 0.6\n', '0 0\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
