import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    a,b=map(int,data.split()); return str(a*b)

print(solve(sys.stdin.read()))
