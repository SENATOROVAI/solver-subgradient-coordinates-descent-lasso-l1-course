"""Ответ 05.3: два первых шага из одной исходной точки."""
import json
import sys
from math import isclose
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import mae, predict


def mae_step(X, y, w, b, rate):
    prediction = predict(X, w, b)
    signs = [(pi > yi) - (pi < yi)
             for yi, pi in zip(y, prediction)]
    n = len(y)
    direction = [sum(signs[i] * X[i][j]
                     for i in range(n)) / n
                 for j in range(len(w))]
    new_w = [wj - rate * gj for wj, gj in zip(w, direction)]
    new_b = b - rate * sum(signs) / n
    return new_w, new_b


X, y = [[1], [2], [3]], [3, 5, 8]
w0, b0 = [2], 0.9
initial = mae(y, predict(X, w0, b0))
records = []
for rate in [0.03, 0.3]:
    w, b = mae_step(X, y, w0, b0, rate)
    prediction = predict(X, w, b)
    loss = mae(y, prediction)
    records.append({"rate": rate, "coef": w, "intercept": b,
                    "prediction": prediction, "mae": loss})
assert records[0]["mae"] < initial < records[1]["mae"]
assert isclose(initial, 1.3 / 3, rel_tol=0, abs_tol=1e-12)
assert isclose(records[0]["mae"], 0.95 / 3, rel_tol=0, abs_tol=1e-12)
assert isclose(records[1]["mae"], 3.2 / 3, rel_tol=0, abs_tol=1e-12)
# В точке с нулевыми остатками даже малый шаг может ухудшить MAE.
corner_w, corner_b = mae_step(X, y, [2], 1, 0.03)
corner_loss = mae(y, predict(X, corner_w, corner_b))
assert corner_loss > 1 / 3
RESULT = {"initial_mae": initial, "records": records,
          "corner_small_step_mae": corner_loss}

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
