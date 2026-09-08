"""Ответ 04.3: промежуток лучших констант на точной сетке."""
import json
import sys
from math import isclose
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import mae, median

records = []
for y in [[1, 2, 4, 9], [1, 2, 4, 90]]:
    grid = [index / 4 for index in range(41)]
    scores = [mae(y, [c] * len(y)) for c in grid]
    minimum = min(scores)
    best = [c for c, score in zip(grid, scores)
            if isclose(score, minimum, rel_tol=0, abs_tol=1e-12)]
    assert isclose(best[0], 2, rel_tol=0, abs_tol=1e-12)
    assert isclose(best[-1], 4, rel_tol=0, abs_tol=1e-12)
    records.append({"y": y, "median": median(y),
                    "minimum": minimum, "best_grid": best,
                    "bounds": [best[0], best[-1]]})
assert isclose(records[0]["minimum"], 2.5, rel_tol=0, abs_tol=1e-12)
assert isclose(records[1]["minimum"], 22.75, rel_tol=0, abs_tol=1e-12)
RESULT = {"records": records}

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
