import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    rows=[s.split() for s in data.splitlines()]; return min(rows,key=lambda r:float(r[1]))[0]

print(solve(sys.stdin.read()))
