import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    lines=data.splitlines(); y=list(map(float,lines[0].split())); p=list(map(float,lines[1].split())); k=float(lines[2]); r=[k*(a-b) for a,b in zip(y,p)]; return f'{sum(abs(v) for v in r)/len(r):.6f} {sum(v*v for v in r)/len(r):.6f}'

print(solve(sys.stdin.read()))
