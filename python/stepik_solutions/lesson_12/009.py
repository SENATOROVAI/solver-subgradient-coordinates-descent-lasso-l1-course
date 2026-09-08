#!/usr/bin/env python3
"""Готовое решение Stepik: урок 12, шаг 009.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    z,lam,q=map(float,data.split()); s=(z>0)-(z<0); return f'{s*max(abs(z)-lam,0)/q:.6f}'

CASES = ['1.4 0.2 2\n', '-1.4 0.2 2\n', '0.1 0.2 3\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
