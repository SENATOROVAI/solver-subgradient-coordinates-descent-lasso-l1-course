#!/usr/bin/env python3
"""Готовое решение Stepik: урок 03, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines=data.splitlines(); y=list(map(float,lines[0].split())); rows=[s.split() for s in lines[1:]]; mae={r[0]:sum(abs(a-b) for a,b in zip(y,map(float,r[1:])))/len(y) for r in rows}; mse={r[0]:sum((a-b)**2 for a,b in zip(y,map(float,r[1:])))/len(y) for r in rows}; return min(mae,key=mae.get)+' '+min(mse,key=mse.get)

CASES = ['0 0 0 0\nA 0 0 0 4\nB 1.2 1.2 1.2 1.2\n', '0 0 0 0\nC 0 0 0 8\nD 3 3 3 3\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
