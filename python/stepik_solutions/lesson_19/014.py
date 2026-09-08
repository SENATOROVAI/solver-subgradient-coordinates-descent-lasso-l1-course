import sys

def fmt(values):
    return ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)

def solve(data):
    lines=data.splitlines(); train=list(map(float,lines[0].split())); test=list(map(float,lines[1].split())); mean=sum(train)/len(train); scale=(sum((x-mean)**2 for x in train)/len(train))**0.5 or 1; return fmt([(x-mean)/scale for x in test])

print(solve(sys.stdin.read()))
