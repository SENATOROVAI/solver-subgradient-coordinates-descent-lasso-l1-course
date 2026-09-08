"""Ручная двойственная проверка: n=2, x=(-1,1), y=(-2,2)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optim_extra import dual_report


X, y = [[-1.], [1.]], [-2., 2.]
# Lasso: из theta=r/n=(-1,1) получаем (-.25,.25).
lasso = dual_report(X, y, [0.], alpha=.5)
assert lasso["theta"] == [-.25, .25]
assert lasso["primal"] == 2. and lasso["dual"] == .875
# EN: lambda1=lambda2=.3; theta=(-1,1) допустима без масштаба.
enet = dual_report(X, y, [0.], alpha=.6, l1_ratio=.5)
expected = 2 - (2 - .3) ** 2 / (2 * .3)
assert abs(enet["dual"] - expected) < 1e-14
print({"lasso_gap": lasso["gap"], "enet_gap": enet["gap"],
       "enet_coefficient_bound": enet["coefficient_bound"]})
