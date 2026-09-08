import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    lines=data.splitlines(); b=float(lines[0]); ws=list(map(float,lines[1].split())); means=list(map(float,lines[2].split())); scales=list(map(float,lines[3].split())); raw=[w/s for w,s in zip(ws,scales)]; intercept=b-sum(w*m for w,m in zip(raw,means)); return fmt([intercept]+raw)

print(solve(sys.stdin.read()))
