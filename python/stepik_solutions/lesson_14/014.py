#!/usr/bin/env python3
"""Готовое решение Stepik: урок 14, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines=data.splitlines(); w1=list(map(float,lines[0].split())); w2=list(map(float,lines[1].split())); rows=[list(map(float,s.split())) for s in lines[2:]]; gaps=[abs(sum(a*b for a,b in zip(x,w1))-sum(a*b for a,b in zip(x,w2))) for x in rows]; return f'{max(gaps):.6f}'

CASES = ['2.8 0\n0 2.8\n-1 -1\n1 1\n', '2.8 0\n0 2.8\n2 2\n1 -1\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
