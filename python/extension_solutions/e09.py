"""Д09.3. Разложение по новым обучающим выборкам; stdlib."""
import random
import sys
from pathlib import Path
from statistics import mean
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extra_utils import finish_extra

rng = random.Random(1707)
R, n, truth, sigma = 2000, 20, 2.0, 1.0
predictions = [mean([truth + rng.gauss(0, sigma) for _ in range(n)])
               for _ in range(R)]
new_y = [truth + rng.gauss(0, sigma) for _ in range(R)]
average = mean(predictions)
bias2 = (average - truth) ** 2
variance = mean([(v - average) ** 2 for v in predictions])
noiseless = mean([(v - truth) ** 2 for v in predictions])
response_mse = mean([(a - b) ** 2 for a, b in zip(predictions, new_y)])
assert abs(noiseless - bias2 - variance) < 1e-12
sample_variance = variance * R / (R - 1)
RESULT = {"R": R, "n": n, "mean_prediction": average,
          "bias2": bias2, "variance_divisor_R": variance,
          "identity_error": abs(noiseless - bias2 - variance),
          "sum_plus_known_noise": noiseless + sigma ** 2,
          "fresh_response_mse": response_mse,
          "excess_if_ddof1": sample_variance - variance,
          "manual": {"mean": 2, "bias2": 0, "variance": 1,
                     "noise": 4, "sum": 5}}
finish_extra("solution_e09", RESULT)
