"""Ответ 01.3: фиксированные данные, случайности нет."""
import json
import sys
from math import isclose
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import predict, residuals

X, y = [[1], [2], [3]], [3, 5, 8]
cases = [(2, 1), (2, 2), (1, 1), (3, 0)]
records = []
for w, b in cases:
    prediction = predict(X, [w], b)
    errors = residuals(y, prediction)
    records.append({"w": w, "b": b, "prediction": prediction,
                    "residuals": errors, "sum": sum(errors)})
for actual, expected in zip(records[0]["prediction"], [3, 5, 7]):
    assert isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
for actual, expected in zip(records[1]["residuals"], [-1, -1, 0]):
    assert isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
RESULT = {"records": records}

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
