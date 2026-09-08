import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    delta,tol=map(float,data.split()); return 'converged' if delta<=tol else 'continue'

print(solve(sys.stdin.read()))
