"""Решение Д23.3: повторить выбор на других общих фолдах."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pure_core import marketing_data
from source_extra import cv_one_se
from extra_utils import finish_extra

X, y, _ = marketing_data(n=120, p=8, seed=620, noise=2.)
alphas = np.geomspace(.005, 1., 16)
result = cv_one_se(X, y, alphas, rho=.7, seed=1620)
chosen = result["one_se_index"]
minimum = result["best_index"]
threshold = result["mean"][minimum] + result["se"][minimum]
assert result["mean"][chosen] <= threshold
assert all(alphas[k] <= alphas[chosen]
           for k, error in enumerate(result["mean"])
           if error <= threshold)
assert np.isclose(threshold, result["threshold"])
RESULT = finish_extra("solution_e23", result)
