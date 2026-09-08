import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    rows=[list(map(float,s.split())) for s in data.splitlines()]; best=min(rows,key=lambda r:r[2]); return f'{best[0]:.6f} {best[1]:.6f}'

print(solve(sys.stdin.read()))
