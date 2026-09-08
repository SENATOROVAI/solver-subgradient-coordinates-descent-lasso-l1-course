import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    v=list(map(float,data.split())); max_iter=int(v[0]); tol=v[1]; history=v[2:]; done=any(abs(b-a)<=tol for a,b in zip(history,history[1:])); return 'converged' if done else 'max_iter'

print(solve(sys.stdin.read()))
