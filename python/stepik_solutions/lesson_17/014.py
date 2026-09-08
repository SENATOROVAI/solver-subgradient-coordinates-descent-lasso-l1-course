import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    lines=data.splitlines(); l1,l2=map(float,lines[0].split()); z=list(map(float,lines[1].split())); q=list(map(float,lines[2].split())); return fmt([((a>0)-(a<0))*max(abs(a)-l1,0)/(b+l2) for a,b in zip(z,q)])

print(solve(sys.stdin.read()))
