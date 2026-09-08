"""Д11.3. Разные масштабы и ненулевые центры prior."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extra_utils import finish_extra
from stat_extra import shifted_laplace_map

n, sigma2 = 20, 4.0
centers, taus, mus = [1.0, 0.4], [0.4, 2.0], [0.0, -0.5]
penalties = [sigma2 / (n * t) for t in taus]
answer = [shifted_laplace_map(c, 1, a, m)
          for c, a, m in zip(centers, penalties, mus)]
grid = [-2 + i / 1000 for i in range(4001)]
grid_answer = []
for c, a, m, w in zip(centers, penalties, mus, answer):
    def objective(v):
        return (v - c) ** 2 / 2 + a * abs(v - m)
    best = min(grid, key=objective)
    grid_answer.append(best)
    assert abs(best - w) < 1e-12
    assert objective(w) <= objective(c) + 1e-12
    assert objective(w) <= objective(m) + 1e-12
alpha80 = 9 / (80 * 0.3)
alpha160 = 9 / (160 * 0.3)
assert abs(alpha160 - alpha80 / 2) < 1e-12
RESULT = {"penalties": penalties, "analytic": answer,
          "grid": grid_answer, "alpha_n80": alpha80,
          "alpha_n160": alpha160,
          "map_mu_1point2": shifted_laplace_map(1, 1, 0.5, 1.2)}
finish_extra("solution_e11", RESULT)
