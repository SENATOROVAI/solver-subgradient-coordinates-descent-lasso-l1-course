import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    lines=data.splitlines(); y=list(map(float,lines[0].split())); rows=[s.split() for s in lines[1:]]; scores={r[0]:sum(abs(a-b) for a,b in zip(y,map(float,r[1:])))/len(y) for r in rows}; return min(scores,key=scores.get)

print(solve(sys.stdin.read()))
