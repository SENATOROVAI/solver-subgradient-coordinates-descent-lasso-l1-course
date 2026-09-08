import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    alpha,rho=map(float,data.split()); return f'{alpha*rho:.6f} {alpha*(1-rho):.6f}'

print(solve(sys.stdin.read()))
