import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    lines=data.splitlines(); a=list(map(float,lines[0].split())); b=list(map(float,lines[1].split())); return f'{sum(v*v for v in a)/len(a):.6f} {sum(v*v for v in b)/len(b):.6f}'

print(solve(sys.stdin.read()))
