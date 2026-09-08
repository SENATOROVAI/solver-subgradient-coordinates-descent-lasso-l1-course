import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    v=list(map(float,data.split())); old,new=v[0],v[1]; return fmt([r-x*(new-old) for x,r in zip(v[2::2],v[3::2])])

print(solve(sys.stdin.read()))
