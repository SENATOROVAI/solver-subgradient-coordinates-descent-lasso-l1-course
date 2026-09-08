#!/usr/bin/env python3
"""Готовое решение Stepik: урок 07, шаг 009.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines=data.splitlines(); a=list(map(float,lines[0].split())); b=list(map(float,lines[1].split())); return f'{sum(v*v for v in a)/len(a):.6f} {sum(v*v for v in b)/len(b):.6f}'

CASES = ['0 0\n2 -2\n', '-1 1\n0 0\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
