#!/usr/bin/env python3
"""Готовое решение Stepik: урок 04, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    y=sorted(map(float,data.split())); n=len(y); c=(y[(n-1)//2]+y[n//2])/2; mae=sum(abs(v-c) for v in y)/n; return f'{c:.6f} {mae:.6f}'

CASES = ['1 2 4 9\n', '1 2 4 90\n', '-1 0 1\n', '8\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
