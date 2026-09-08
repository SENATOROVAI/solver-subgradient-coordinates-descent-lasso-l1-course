"""Д17.3. Проверяем область действия границы после масштаба и знака."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
from pure_core import coordinate_descent, scale_fit, scale_transform
from geometry_extra import grouping_pair
from extra_utils import finish_extra

raw = [[-1.1, -1.], [-.9, -1.], [.9, 1.], [1.1, 1.]]
X = scale_transform(raw, scale_fit(raw))
y = [-3., -3., 3., 3.]
records = []
for multiplier in [1., 2., -1.]:
    changed = [[row[0], multiplier * row[1]] for row in X]
    fit = coordinate_descent(changed, y, .5, .4, tol=1e-11)
    assert fit['converged']
    report = grouping_pair(changed, y, fit['coef'], 0, 1, .3)
    if multiplier > 0:
        assert report['valid']
        assert report['difference'] <= report['direct_bound'] + 1e-8
    if multiplier == 2.:
        assert report['correlation_bound'] is None
    if multiplier == -1.:
        assert not report['valid']
        oriented = [fit['coef'][0], -fit['coef'][1]]
        report['after_orientation'] = grouping_pair(X, y, oriented,
                                                    0, 1, .3)
        assert report['after_orientation']['valid']
    records.append({'multiplier': multiplier, 'report': report})
RESULT = {'manual_bound': .4 / .2 * math.sqrt(2 * (1 - .995)),
          'weight_of_four_copies': 2.8 / 4.8, 'experiments': records}
finish_extra('solution_e17', RESULT)
