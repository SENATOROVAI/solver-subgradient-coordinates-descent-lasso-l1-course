"""Самостоятельный пример баланса KKT в исходных координатах."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optim_extra import alpha_max, kkt_report


X = [[9., 4.], [11., 4.]]
y = [18., 22.]
# r=[-.5,.5]; X^T r/n=[.5,0]; mean(r)=0.
report = kkt_report(X, y, w=[1.5, 0.], b=5., alpha=.5)
assert report["coordinate_violations"] == [0., 0.]
assert report["intercept_violation"] == 0.
assert alpha_max(X, y) == 2.
print(report)
