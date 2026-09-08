"""11.3: мягкий порог против независимого перебора критерия."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import soft_threshold

alpha = 0.8
records = []
for z in [-1.8, 0.4, 1.8]:
    # q=1: отброшенная константа не влияет на минимум.
    def criterion(w):
        return 0.5 * (w - z) ** 2 + alpha * abs(w)

    grid = [index / 1000.0 for index in range(-3000, 3001)]
    grid_weight = min(grid, key=criterion)
    weight = soft_threshold(z, alpha)
    assert abs(grid_weight - weight) < 1e-10
    records.append({"z": z, "soft": weight, "grid": grid_weight,
                    "criterion": criterion(weight)})
for z in [-2.0, -0.8, 0.0, 0.8, 2.0]:
    assert abs(soft_threshold(z, 0.0) - z) < 1e-12
assert abs(soft_threshold(-alpha, alpha)) < 1e-12
assert abs(soft_threshold(alpha, alpha)) < 1e-12
try:
    soft_threshold(1.0, -0.1)
except ValueError:
    rejected_negative = True
else:
    rejected_negative = False
assert rejected_negative
RESULT = {"records": records, "negative_threshold_rejected": True}

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
