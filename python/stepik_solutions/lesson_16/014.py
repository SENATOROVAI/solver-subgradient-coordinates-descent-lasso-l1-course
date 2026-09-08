import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    l1,l2=map(float,data.split()); alpha=l1+l2; rho=l1/alpha if alpha else 0; return f'{alpha:.6f} {rho:.6f}'

print(solve(sys.stdin.read()))
