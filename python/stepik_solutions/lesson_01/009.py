import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    lines=data.splitlines(); w,b=map(float,lines[0].split()); xs=list(map(float,lines[1].split())); ys=list(map(float,lines[2].split())); p=[w*x+b for x in xs]; r=[y-q for y,q in zip(ys,p)]; return fmt(p)+'\n'+fmt(r)

print(solve(sys.stdin.read()))
