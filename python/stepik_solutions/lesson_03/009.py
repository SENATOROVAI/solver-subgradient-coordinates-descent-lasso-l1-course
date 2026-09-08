#!/usr/bin/env python3
"""Готовое решение Stepik: урок 03, шаг 009.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines=data.splitlines(); metric=lines[0].strip(); y=list(map(float,lines[1].split())); rows=[s.split() for s in lines[2:]]; score={r[0]:(sum(abs(a-b) if metric=='MAE' else (a-b)**2 for a,b in zip(y,map(float,r[1:])))/len(y)) for r in rows}; return ' '.join(sorted(score,key=score.get))

CASES = ['MAE\n0 0 0 0\nA 0 0 0 4\nB 1.2 1.2 1.2 1.2\nC 0 0 0 8\nD 3 3 3 3\n', 'MSE\n0 0 0 0\nA 0 0 0 4\nB 1.2 1.2 1.2 1.2\nC 0 0 0 8\nD 3 3 3 3\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
