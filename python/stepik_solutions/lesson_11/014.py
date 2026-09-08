#!/usr/bin/env python3
"""Готовое решение Stepik: урок 11, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    v=list(map(float,data.split())); t=v[0]; return fmt([((z>0)-(z<0))*max(abs(z)-t,0) for z in v[1:]])

CASES = ['0.8 -2 -0.8 0 0.8 2\n', '0 1 -3 0\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
