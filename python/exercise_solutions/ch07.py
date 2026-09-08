"""Ответ 07.3: отдельная оценка на обучении и проверке."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import mae, mse

train_y, validation_y = [2, 4], [3, 3]
predictions = {"flexible": {"train": [2, 4], "validation": [1, 5]},
               "constant": {"train": [3, 3], "validation": [3, 3]}}
records = {}
for name, p in predictions.items():
    records[name] = {
        "train_mae": mae(train_y, p["train"]),
        "train_mse": mse(train_y, p["train"]),
        "validation_mae": mae(validation_y, p["validation"]),
        "validation_mse": mse(validation_y, p["validation"]),
    }
train_winner = min(records, key=lambda k: records[k]["train_mse"])
val_winner = min(records, key=lambda k: records[k]["validation_mse"])
assert train_winner == "flexible"
assert val_winner == "constant"
assert train_winner != val_winner
RESULT = {"records": records, "train_winner": train_winner,
          "validation_winner": val_winner}

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
