#!/usr/bin/env python3
"""Готовое решение Stepik: урок 19, шаг 014.

Программа читает вход из stdin и печатает ответ в stdout. Алгоритм получен
из проверенного авторского кода соответствующего локального .step.
"""

def fmt(values):
    cleaned = [0.0 if isinstance(v, float) and abs(v) < 5e-13 else v for v in values]
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in cleaned)

def solve(data):
    lines=data.splitlines(); train=list(map(float,lines[0].split())); test=list(map(float,lines[1].split())); mean=sum(train)/len(train); scale=(sum((x-mean)**2 for x in train)/len(train))**0.5 or 1; return fmt([(x-mean)/scale for x in test])

CASES = ['0 2\n100\n', '1 1\n1 3\n', '-1 1\n0 2\n']

def generate():
    return [(case, solve(case)) for case in CASES]

def check(reply, clue):
    return reply.strip() == clue.strip()

if __name__ == "__main__":
    import sys

    result = solve(sys.stdin.read())
    if result is not None:
        sys.stdout.write(str(result))
