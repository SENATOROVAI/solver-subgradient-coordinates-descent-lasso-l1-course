import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    n,share=map(float,data.split()); test=round(n*share); return f'{int(n-test)} {int(test)}'

print(solve(sys.stdin.read()))
