import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    v=list(map(float,data.split())); alpha=v[0]; return fmt([(1 if z>0 else -1 if z<0 else 0)*max(abs(z)-alpha,0) for z in v[1:]])

print(solve(sys.stdin.read()))
