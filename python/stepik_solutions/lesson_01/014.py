import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    lines=data.splitlines(); xs=list(map(float,lines[0].split())); ys=list(map(float,lines[1].split())); models=[tuple(map(float,s.split())) for s in lines[2:]]; scores=[sum(abs(y-(w*x+b)) for x,y in zip(xs,ys)) for w,b in models]; k=min(range(len(scores)),key=lambda i:scores[i]); return f'{k+1} {scores[k]:.6f}'

print(solve(sys.stdin.read()))
