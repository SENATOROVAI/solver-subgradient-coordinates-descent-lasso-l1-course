"""Ответ 02.3: проверка единиц на фиксированных данных."""
import json
import sys
from math import isclose
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import mae, mse, residuals

y, prediction = [1, 4, 8], [3, 4, 6]
base_mae, base_mse = mae(y, prediction), mse(y, prediction)
records = []
for factor in [0.1, 1, 10]:
    scaled_y = [factor * value for value in y]
    scaled_prediction = [factor * value for value in prediction]
    a = mae(scaled_y, scaled_prediction)
    q = mse(scaled_y, scaled_prediction)
    signed = sum(residuals(scaled_y, scaled_prediction)) / len(y)
    assert isclose(a, abs(factor) * base_mae, rel_tol=0, abs_tol=1e-12)
    assert isclose(q, factor ** 2 * base_mse, rel_tol=0, abs_tol=1e-10)
    assert abs(signed) < 1e-12
    records.append({"factor": factor, "signed_mean": signed,
                    "mae": a, "mse": q})
RESULT = {"base_mae": base_mae, "base_mse": base_mse,
          "records": records}

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
