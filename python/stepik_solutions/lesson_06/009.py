import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    r=list(map(float,data.split())); return f'{sum(abs(v) for v in r)/len(r):.6f} {sum(v*v for v in r)/len(r):.6f}'

print(solve(sys.stdin.read()))
