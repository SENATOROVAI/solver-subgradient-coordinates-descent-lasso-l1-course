import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    v=list(map(float,data.split())); x1,x2=v[:2]; a,b,c,d=v[2:]; return f'{a*x1+b*x2:.6f} {c*x1+d*x2:.6f}'

print(solve(sys.stdin.read()))
