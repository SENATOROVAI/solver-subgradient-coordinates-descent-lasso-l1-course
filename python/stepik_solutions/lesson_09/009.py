import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    v=list(map(float,data.split())); alpha=v[0]; return f'{alpha*sum(abs(x) for x in v[1:]):.6f}'

print(solve(sys.stdin.read()))
