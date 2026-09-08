from pure_core import (_vector)

def median(values):
    values = sorted(_vector(values, "values"))
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return values[middle - 1] / 2 + values[middle] / 2
