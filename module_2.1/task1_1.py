def main():
    n = int(input())
    values = list(map(float, input().split()))
    values.sort()

    if n % 2 == 1:
        c = values[n // 2]
    else:
        c = (values[n // 2 - 1] + values[n // 2]) / 2.0

    mae = sum(abs(v - c) for v in values) / n
    print("{:.6f} {:.6f}".format(c, mae))
