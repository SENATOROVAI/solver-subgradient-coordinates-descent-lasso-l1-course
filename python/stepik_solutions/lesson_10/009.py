import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    v=list(map(float,data.split())); t=v[0]; return 'YES' if sum(abs(x) for x in v[1:])<=t+1e-12 else 'NO'

print(solve(sys.stdin.read()))
