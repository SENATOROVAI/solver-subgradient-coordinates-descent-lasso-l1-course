#!/usr/bin/env python3
"""Готовое решение Stepik: урок 05, шаг 009.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines=data.splitlines(); x=list(map(float,lines[0].split())); y=list(map(float,lines[1].split())); w,b,rate=map(float,lines[2].split()); s=[(w*a+b>v)-(w*a+b<v) for a,v in zip(x,y)]; gw=sum(q*a for q,a in zip(s,x))/len(x); gb=sum(s)/len(s); return f'{w-rate*gw:.6f} {b-rate*gb:.6f}'

CASES = ['1 2 3\n3 5 8\n2 1 0.1\n', '1 2 3\n3 5 8\n2 0.9 0.03\n', '0\n0\n1 2 0.5\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
