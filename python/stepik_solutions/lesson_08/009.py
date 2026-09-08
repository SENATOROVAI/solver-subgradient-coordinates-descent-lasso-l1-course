import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    v=list(map(float,data.split())); mean,scale=v[0],v[1]; return fmt([(x-mean)/scale for x in v[2:]])

print(solve(sys.stdin.read()))
