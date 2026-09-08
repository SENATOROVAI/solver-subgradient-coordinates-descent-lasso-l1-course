"""Ответ 03.3: две метрики дают два рейтинга."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import mae, mse

y = [0, 0, 0, 0]
predictions = {"A": [0, 0, 0, 4], "B": [1.2] * 4,
               "C": [0, 0, 0, 8], "D": [3] * 4}
scores = {name: {"mae": mae(y, p), "mse": mse(y, p)}
          for name, p in predictions.items()}
mae_order = sorted(scores, key=lambda name: scores[name]["mae"])
mse_order = sorted(scores, key=lambda name: scores[name]["mse"])
assert mae_order.index("A") < mae_order.index("B")
assert mse_order.index("B") < mse_order.index("A")
assert mae_order.index("C") < mae_order.index("D")
assert mse_order.index("D") < mse_order.index("C")
RESULT = {"scores": scores, "mae_order": mae_order,
          "mse_order": mse_order}

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
