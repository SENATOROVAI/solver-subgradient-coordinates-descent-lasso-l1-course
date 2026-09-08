#!/usr/bin/env python3
"""Готовое решение Stepik: урок 07, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines=data.splitlines(); rows=[s.split() for s in lines]; scores={r[0]:sum(float(v)**2 for v in r[1:])/len(r[1:]) for r in rows}; return min(scores,key=scores.get)

CASES = ['flexible 2 -2\nconstant 0 0\n', 'small 1 1\nlarge 0 3\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
