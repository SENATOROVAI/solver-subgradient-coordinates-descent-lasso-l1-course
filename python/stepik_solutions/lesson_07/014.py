import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    lines=data.splitlines(); rows=[s.split() for s in lines]; scores={r[0]:sum(float(v)**2 for v in r[1:])/len(r[1:]) for r in rows}; return min(scores,key=scores.get)

print(solve(sys.stdin.read()))
