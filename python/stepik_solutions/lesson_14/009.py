#!/usr/bin/env python3
"""Готовое решение Stepik: урок 14, шаг 009.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    v=list(map(float,data.split())); x1,x2=v[:2]; a,b,c,d=v[2:]; return f'{a*x1+b*x2:.6f} {c*x1+d*x2:.6f}'

CASES = ['2 2 2.8 0 0 2.8\n', '1 -1 2.8 0 0 2.8\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
