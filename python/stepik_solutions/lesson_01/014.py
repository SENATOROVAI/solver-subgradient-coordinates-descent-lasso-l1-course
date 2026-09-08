#!/usr/bin/env python3
"""Готовое решение Stepik: урок 01, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines=data.splitlines(); xs=list(map(float,lines[0].split())); ys=list(map(float,lines[1].split())); models=[tuple(map(float,s.split())) for s in lines[2:]]; scores=[sum(abs(y-(w*x+b)) for x,y in zip(xs,ys)) for w,b in models]; k=min(range(len(scores)),key=lambda i:scores[i]); return f'{k+1} {scores[k]:.6f}'

CASES = ['1 2 3\n3 5 8\n2 1\n2 2\n1 1\n', '0 1\n1 1\n0 1\n1 0\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
