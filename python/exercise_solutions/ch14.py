"""14.3: разные веса при одинаковых столбцах и прогнозах."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import lasso_cd, predict

X = [[-1.0, -1.0], [1.0, 1.0]]
y = [-3.0, 3.0]
models = [lasso_cd(X, y, alpha=0.2, w0=start, tol=1e-12)
          for start in [[0.0, 0.0], [0.0, 3.0]]]
assert all(model["converged"] for model in models)
a, b = models
weight_gap = max(abs(x - z) for x, z in zip(a["coef"], b["coef"]))
assert weight_gap > 2.7
pa = predict(X, a["coef"], a["intercept"])
pb = predict(X, b["coef"], b["intercept"])
assert max(abs(x - z) for x, z in zip(pa, pb)) < 1e-10
assert abs(a["objective"] - b["objective"]) < 1e-10
assert abs(a["objective"] - 0.58) < 1e-10
probe = [[2.0, 2.0], [1.0, -1.0]]
probe_predictions = [predict(probe, m["coef"], m["intercept"])
                     for m in models]
assert abs(probe_predictions[0][0] - probe_predictions[1][0]) < 1e-10
assert abs(probe_predictions[0][1] - probe_predictions[1][1]) > 5.5
RESULT = {"coef": [m["coef"] for m in models],
          "objective": [m["objective"] for m in models],
          "training_prediction": [pa, pb], "probe": probe,
          "probe_predictions": probe_predictions}

if __name__ == "__main__":
    print(json.dumps(RESULT, ensure_ascii=False, indent=2))
