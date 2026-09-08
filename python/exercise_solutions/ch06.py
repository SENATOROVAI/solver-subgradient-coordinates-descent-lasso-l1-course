"""Ответ 06.3: два явно разных вмешательства в таблицу."""
import json
import sys
from math import isclose
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import mae, mse, predict


def score(X, y, w, b):
    p = predict(X, [w], b)
    return {"prediction": p, "mae": mae(y, p), "mse": mse(y, p)}


y_outlier_X, y_outlier_y = [[-1], [0], [1]], [-1, 9, 1]
y_case = {"original_line": score(y_outlier_X, y_outlier_y, 1, 0),
          "shifted_line": score(y_outlier_X, y_outlier_y, 1, 3)}
large_X, large_y = [[-1], [0], [1], [10]], [-1, 0, 1, -10]
x_case = {"original_line": score(large_X, large_y, 1, 0),
          "constant_zero": score(large_X, large_y, 0, 0)}
assert y_case["original_line"]["mae"] < y_case["shifted_line"]["mae"]
assert y_case["original_line"]["mse"] > y_case["shifted_line"]["mse"]
assert x_case["constant_zero"]["mae"] < x_case["original_line"]["mae"]
assert isclose(x_case["original_line"]["mse"], 100,
                   rel_tol=0, abs_tol=1e-12)
assert isclose(x_case["constant_zero"]["mse"], 25.5,
                   rel_tol=0, abs_tol=1e-12)
RESULT = {"y_outlier": y_case, "large_feature": x_case}

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
