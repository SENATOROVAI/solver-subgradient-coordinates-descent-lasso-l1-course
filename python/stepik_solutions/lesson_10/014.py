#!/usr/bin/env python3
"""Готовое решение Stepik: урок 10, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines=data.splitlines(); t=float(lines[0]); rows=[list(map(float,s.split())) for s in lines[1:]]; ok=[r for r in rows if abs(r[0])+abs(r[1])<=t+1e-12]; best=min(ok,key=lambda r:r[2]); return fmt(best[:2]+[best[2]])

CASES = ['1.2\n1.2 0 0.445\n0.6 0.6 0.7\n1 1 0.1\n', '1\n0 0 2\n1 0 1\n0 2 0\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
