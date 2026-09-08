import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    lines=data.splitlines(); w1=list(map(float,lines[0].split())); w2=list(map(float,lines[1].split())); rows=[list(map(float,s.split())) for s in lines[2:]]; gaps=[abs(sum(a*b for a,b in zip(x,w1))-sum(a*b for a,b in zip(x,w2))) for x in rows]; return f'{max(gaps):.6f}'

print(solve(sys.stdin.read()))
