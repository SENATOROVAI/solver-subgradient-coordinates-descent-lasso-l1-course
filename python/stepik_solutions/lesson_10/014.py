import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    lines=data.splitlines(); t=float(lines[0]); rows=[list(map(float,s.split())) for s in lines[1:]]; ok=[r for r in rows if abs(r[0])+abs(r[1])<=t+1e-12]; best=min(ok,key=lambda r:r[2]); return fmt(best[:2]+[best[2]])

print(solve(sys.stdin.read()))
