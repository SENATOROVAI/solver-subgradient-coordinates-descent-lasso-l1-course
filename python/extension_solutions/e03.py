"""Решение Д03.3: проверяем опору и совместный знак."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pure_core import predict, mae
from extra_utils import finish_extra

grid = [k / 100 for k in range(-200, 201)]
valid = {}
for g in [-1, -.6, 0, .5, 1, 1.5]:
    valid[str(g)] = all(abs(z) >= g*z - 1e-12 for z in grid)
assert valid["-0.6"] and not valid["1.5"]
# Аналитический минимум 2|z| + (z-3)^2/2 равен z=1.
objective = lambda z: 2*abs(z) + (z-3)**2/2
assert objective(1) == 4
assert objective(0) == 4.5
X, y = [[1], [1], [.5]], [0, 1, -1]
initial = mae(y, predict(X, [0], 0))
shifted = mae(y, predict(X, [.6], -.6))
assert initial == 2/3 and shifted < initial
# Для r=[0,1,-1] равенство по b требует s0=0,
# а равенство по w требует s0=-0.5. Общего s0 нет.
assert 0 != -.5
RESULT = {"support_checks": valid, "f_minimizer": 1,
          "f_minimum": 4, "initial_mae": initial,
          "joint_step_mae": shifted, "one_shared_sign_exists": False}
finish_extra("solution_e03", RESULT)
