import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    y=sorted(map(float,data.split())); n=len(y); return fmt([y[(n-1)//2],y[n//2]])

print(solve(sys.stdin.read()))
