import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    lines=data.splitlines(); metric=lines[0].strip(); y=list(map(float,lines[1].split())); rows=[s.split() for s in lines[2:]]; score={r[0]:(sum(abs(a-b) if metric=='MAE' else (a-b)**2 for a,b in zip(y,map(float,r[1:])))/len(y)) for r in rows}; return ' '.join(sorted(score,key=score.get))

print(solve(sys.stdin.read()))
