import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    v=list(map(float,data.split())); t=v[0]; return fmt([((z>0)-(z<0))*max(abs(z)-t,0) for z in v[1:]])

print(solve(sys.stdin.read()))
