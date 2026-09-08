import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    z,lam,q=map(float,data.split()); s=(z>0)-(z<0); return f'{s*max(abs(z)-lam,0)/q:.6f}'

print(solve(sys.stdin.read()))
