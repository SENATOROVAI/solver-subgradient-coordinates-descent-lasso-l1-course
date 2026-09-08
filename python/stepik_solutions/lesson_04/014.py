import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    y=sorted(map(float,data.split())); n=len(y); c=(y[(n-1)//2]+y[n//2])/2; mae=sum(abs(v-c) for v in y)/n; return f'{c:.6f} {mae:.6f}'

print(solve(sys.stdin.read()))
