import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    n,lam=map(float,data.split()); return f'{n*lam:.6f}'

print(solve(sys.stdin.read()))
