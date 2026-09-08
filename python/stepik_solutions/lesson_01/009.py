#!/usr/bin/env python3
"""Готовое решение Stepik: урок 01, шаг 009.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines=data.splitlines(); w,b=map(float,lines[0].split()); xs=list(map(float,lines[1].split())); ys=list(map(float,lines[2].split())); p=[w*x+b for x in xs]; r=[y-q for y,q in zip(ys,p)]; return fmt(p)+'\n'+fmt(r)

CASES = ['2 1\n1 2 3\n3 5 8\n', '2 2\n1 2 3\n3 5 8\n', '-1 0\n-2 0 2\n2 0 -2\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
