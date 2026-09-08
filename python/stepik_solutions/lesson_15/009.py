import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    v=list(map(float,data.split())); lam=v[0]; return fmt([z/(1+lam) for z in v[1:]])

print(solve(sys.stdin.read()))
