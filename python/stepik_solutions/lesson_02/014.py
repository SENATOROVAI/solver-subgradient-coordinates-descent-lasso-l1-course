#!/usr/bin/env python3
"""Готовое решение Stepik: урок 02, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines=data.splitlines(); y=list(map(float,lines[0].split())); p=list(map(float,lines[1].split())); k=float(lines[2]); r=[k*(a-b) for a,b in zip(y,p)]; return f'{sum(abs(v) for v in r)/len(r):.6f} {sum(v*v for v in r)/len(r):.6f}'

CASES = ['1 4 8\n3 4 6\n10\n', '1 4 8\n3 4 6\n0.1\n', '1 1\n1 1\n-3\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
