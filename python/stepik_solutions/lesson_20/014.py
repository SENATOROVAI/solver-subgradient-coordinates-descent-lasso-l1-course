import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    lines=data.splitlines(); threshold=float(lines[0]); rows=[list(map(float,s.split())) for s in lines[1:]]; n=len(rows); return fmt([sum(abs(r[j])>threshold for r in rows)/n for j in range(len(rows[0]))])

print(solve(sys.stdin.read()))
