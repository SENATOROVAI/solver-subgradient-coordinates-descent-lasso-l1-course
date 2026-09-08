import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    lines=data.splitlines(); x=list(map(float,lines[0].split())); y=list(map(float,lines[1].split())); w,b,rate=map(float,lines[2].split()); s=[(w*a+b>v)-(w*a+b<v) for a,v in zip(x,y)]; gw=sum(q*a for q,a in zip(s,x))/len(x); gb=sum(s)/len(s); return f'{w-rate*gw:.6f} {b-rate*gb:.6f}'

print(solve(sys.stdin.read()))
