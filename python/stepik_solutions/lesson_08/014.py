#!/usr/bin/env python3
"""Готовое решение Stepik: урок 08, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines=data.splitlines(); b=float(lines[0]); ws=list(map(float,lines[1].split())); means=list(map(float,lines[2].split())); scales=list(map(float,lines[3].split())); raw=[w/s for w,s in zip(ws,scales)]; intercept=b-sum(w*m for w,m in zip(raw,means)); return fmt([intercept]+raw)

CASES = ['10\n6 -2\n10 5\n2 1\n', '0\n3\n4\n2\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
