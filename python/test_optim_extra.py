"""Исполняемые проверки; NumPy/sklearn только в библиотечном мосте.

Запуск: python code/test_optim_extra.py. Последняя строка — JSON-отчёт.
"""

import ast
import contextlib
import io
import json
import math
from pathlib import Path
import random
import unittest

from optim_extra import (
    alpha_max, dual_report, ista, kkt_report, lasso_subgradient,
)
from pure_core import coordinate_descent, objective, predict


REPORT = {"module": "optim_extra", "comparisons": []}


def data(seed=13, n=35, p=4):
    rng = random.Random(seed)
    X = [[rng.gauss(0, 1) + j for j in range(p)] for _ in range(n)]
    beta = [2.0, -1.0, 0.0, 0.6][:p]
    y = [3 + sum(v * b for v, b in zip(row, beta))
         + .1 * rng.gauss(0, 1) for row in X]
    return X, y


class OptimizationTests(unittest.TestCase):
    def assert_vector(self, left, right, tol=1e-8):
        self.assertEqual(len(left), len(right))
        self.assertLessEqual(max(abs(a - b) for a, b in zip(left, right)),
                             tol)

    def test_manual_kkt_and_alpha_max(self):
        X, y = [[9., 4.], [11., 4.]], [18., 22.]
        self.assertEqual(alpha_max(X, y), 2.0)
        self.assertEqual(alpha_max(X, y, l1_ratio=.25), 8.0)
        self.assertEqual(alpha_max(X, y, fit_intercept=False), 202.0)
        report = kkt_report(X, y, [1.5, 0], b=5, alpha=.5)
        self.assert_vector(report["correlations"], [.5, 0])
        self.assertLess(report["max_violation"], 1e-14)
        report = kkt_report(X, y, [0, 0], b=19, alpha=.5)
        self.assertEqual(report["intercept_violation"], 1.)
        self.assert_vector(report["coordinate_violations"], [11.5, 3.5])
        for rho in [.2, 1.0]:
            cutoff = alpha_max(X, y, l1_ratio=rho)
            solution = ista(X, y, alpha=cutoff, l1_ratio=rho)
            self.assertEqual(solution["coef"], [0., 0.])
            below = ista(X, y, alpha=.99 * cutoff, l1_ratio=rho)
            self.assertGreater(below["coef"][0], 0)

    def test_manual_one_coordinate_duals(self):
        X, y = [[-1.], [1.]], [-2., 2.]
        for rho in [0., .5, 1.]:
            l1, l2 = .6 * rho, .6 * (1 - rho)
            expected = (2 - l1) / (1 + l2)
            fit = ista(X, y, alpha=.6, l1_ratio=rho)
            self.assertAlmostEqual(fit["coef"][0], expected)
            report = dual_report(X, y, [expected], alpha=.6,
                                 l1_ratio=rho)
            primal = (2 - expected) ** 2 / 2 + l1 * expected
            primal += l2 * expected ** 2 / 2
            self.assertAlmostEqual(report["primal"], primal)
            self.assertAlmostEqual(report["dual"], primal)
            self.assertLess(abs(report["gap"]), 1e-14)
        report = dual_report(X, y, [0], alpha=.6, l1_ratio=.5)
        self.assertAlmostEqual(report["dual"], 2 - 1.7 ** 2 / .6)
        report = dual_report(X, y, [0], alpha=.5)
        self.assert_vector(report["theta"], [-.25, .25])
        self.assertAlmostEqual(report["dual"], .875)
        self.assertAlmostEqual(report["gap"], 1.125)

    def test_ista_split_paths_and_costs(self):
        X, y = data()
        for rho in [0., .3, 1.]:
            kw = dict(alpha=.4, l1_ratio=rho, max_iter=17, tol=0,
                      w0=[.2, -.5, 1., .3])
            a = ista(X, y, split="smooth_l2", **kw)
            b = ista(X, y, split="prox_all", **kw)
            means = [sum(c) / len(X) for c in zip(*X)]
            expected_L0 = sum((v - m) ** 2 for row in X
                              for v, m in zip(row, means)) / len(X)
            self.assertAlmostEqual(a["L0"], expected_L0)
            self.assertAlmostEqual(a["step"], 1 / (a["L0"]
                                   + .4 * (1 - rho)))
            self.assertAlmostEqual(b["step"], 1 / b["L0"])
            self.assertEqual(a["n_iter"], 17)
            for wa, wb in zip(a["coef_history"], b["coef_history"]):
                self.assert_vector(wa, wb, 2e-14)
            for i, w in enumerate(a["coef_history"]):
                intercept = sum(y) / len(y) - sum(
                    m * v for m, v in zip(means, w))
                loss = objective(X, y, w, intercept, .4, rho)
                self.assertAlmostEqual(a["history"][i], loss)
            self.assertTrue(all(later <= earlier + 1e-13 for
                                earlier, later in zip(a["history"],
                                                     a["history"][1:])))

    def test_constant_zero_and_single_row(self):
        for X, y in [([[7., 0.]] * 3, [1., 2., 6.]),
                     ([[7., 0.]], [4.])]:
            self.assertEqual(alpha_max(X, y), 0.)
            for split in ["smooth_l2", "prox_all"]:
                for rho in [0., .4, 1.]:
                    result = ista(X, y, alpha=.5, l1_ratio=rho,
                                  w0=[3, -2], split=split)
                    self.assertEqual(result["coef"], [0., 0.])
                    self.assertEqual(result["n_iter"], 1)
                    self.assertIsNone(result["step"])
                    self.assertTrue(result["converged"])
                    report = dual_report(X, y, result["coef"],
                                         result["intercept"], .5, rho)
                    self.assertLess(abs(report["gap"]), 1e-12)
                result = ista(X, y, alpha=0, w0=[3, -2], split=split)
                self.assertEqual(result["coef"], [3, -2])
                self.assertEqual(result["n_iter"], 0)
                self.assertTrue(result["converged"])
        X = [[-1., 0., 3.], [1., 0., 3.]]
        fit = ista(X, [-2., 2.], alpha=.4, l1_ratio=.5)
        self.assert_vector(fit["coef"], [1.5, 0., 0.])

    def test_truthful_budget_and_start(self):
        X, y = data()
        start = ista(X, y, max_iter=0)
        self.assertEqual(start["n_iter"], 0)
        self.assertFalse(start["converged"])
        self.assertEqual(start["status"], "max_iter")
        self.assertEqual(len(start["history"]), 1)
        one = ista(X, y, max_iter=1, tol=1e-14)
        self.assertEqual(one["n_iter"], 1)
        self.assertFalse(one["converged"])
        solved = ista(X, y, tol=1e-9)
        self.assertTrue(solved["converged"])
        self.assertEqual(solved["status"], "converged")
        self.assertLessEqual(solved["optimality_error"], 1e-9)
        start = ista(X, y, w0=solved["coef"], tol=1e-9)
        self.assertEqual(start["n_iter"], 0)

    def test_dual_intercept_projection(self):
        X, y = [[10.], [12.], [15.]], [11., 9., 17.]
        for rho in [0., .4, 1.]:
            report = dual_report(X, y, [.7], b=-8., alpha=.6,
                                 l1_ratio=rho)
            self.assertLess(report["feasibility_error"], 1e-12)
            self.assertAlmostEqual(sum(report["theta"]), 0.)
            self.assertGreaterEqual(report["gap"], 0.)
            fit = ista(X, y, alpha=.6, l1_ratio=rho, tol=1e-10)
            self.assertLessEqual(report["dual"], fit["objective"] + 1e-10)
        unpenalized = dual_report(X, y, [.7], b=-8., alpha=0.)
        self.assertEqual(unpenalized["theta"], [0.] * 3)
        self.assertEqual(unpenalized["dual"], 0.)
        self.assertEqual(unpenalized["gap"], unpenalized["primal"])
        self.assertEqual(unpenalized["method"],
                         "unpenalized_zero_certificate")

    def test_negative_roundoff_is_retained(self):
        alpha, rho = .27738484578367845, .8404850621984881
        X = [[-1.], [1.]]
        y = [-3.821235348693304, 3.821235348693304]
        fit = ista(X, y, alpha, rho, tol=1e-13)
        d = dual_report(X, y, fit["coef"], fit["intercept"], alpha, rho)
        self.assertEqual(d["gap"], d["primal"] - d["dual"])
        self.assertLess(d["gap"], 0.)
        self.assertGreater(d["gap"], -1e-14)
        self.assertIsNone(d["coefficient_bound"])

    def test_coefficient_bound_and_feasible_lasso(self):
        X, y = data()
        for rho in [0., .35, 1.]:
            fit = ista(X, y, alpha=.5, l1_ratio=rho, tol=1e-11)
            for w in [[0.] * 4, [2., -1., 1., -.3], fit["coef"]]:
                b = sum(y) / len(y) - sum(
                    sum(c) / len(y) * v for c, v in zip(zip(*X), w))
                d = dual_report(X, y, w, b, .5, rho)
                self.assertGreaterEqual(d["gap"], -1e-12)
                self.assertLess(d["feasibility_error"], 1e-12)
                self.assertLessEqual(d["dual"], fit["objective"] + 1e-11)
                if rho < 1 and d["coefficient_bound"] is not None:
                    distance = math.sqrt(sum((a - b) ** 2 for a, b in
                                             zip(w, fit["coef"])))
                    self.assertLessEqual(distance,
                                         d["coefficient_bound"] + 1e-8)

    def test_subgradient_actual_history(self):
        X, y = [[-1.], [1.]], [-2., 2.]
        fit = lasso_subgradient(X, y, alpha=.5, max_iter=3,
                                step=1., decay=0.)
        self.assertEqual(fit["coef_history"], [[0.], [2.], [1.5], [1.5]])
        self.assertEqual(fit["history"], [2., 1., .875, .875])
        self.assertEqual(fit["best_iteration"], 2)
        self.assertEqual(fit["n_iter"], 3)
        self.assertEqual(fit["status"], "fixed_budget")
        self.assertTrue(fit["converged"])
        oscillation = lasso_subgradient([[-1.], [1.]], [-.2, .2],
                                        alpha=.5, max_iter=10,
                                        step=1., decay=0.)
        self.assertTrue(any(b > a for a, b in zip(
            oscillation["history"], oscillation["history"][1:])))
        self.assertEqual(oscillation["best_coef"], [0.])
        self.assertFalse(oscillation["converged"])
        self.assertNotEqual(oscillation["coef"], [0.])
        self.assertEqual(len(oscillation["coef_history"]), 11)
        zero = lasso_subgradient(X, y, max_iter=0)
        self.assertEqual(zero["n_iter"], 0)
        self.assertEqual(len(zero["history"]), 1)

    def test_no_intercept_and_all_inactive(self):
        X, y = data(n=12, p=2)
        cutoff = alpha_max(X, y, fit_intercept=False)
        zero = ista(X, y, alpha=cutoff, fit_intercept=False)
        self.assertEqual(zero["coef"], [0., 0.])
        self.assertEqual(zero["intercept"], 0.)
        report = dual_report(X, y, [0, 0], alpha=cutoff,
                             fit_intercept=False)
        self.assertLess(abs(report["gap"]), 1e-11)
        self.assertLess(report["feasibility_error"], 1e-11)

    def test_invalid_inputs(self):
        X, y = [[1.], [2.]], [2., 3.]
        bad_data = [([], []), ([[]], [1]), ([[1], [1, 2]], y),
                    (X, [1]), ([[math.nan]], [1]),
                    ([[1]], [math.inf]), ([[True]], [1])]
        for bad_X, bad_y in bad_data:
            for call in [lambda: ista(bad_X, bad_y),
                         lambda: alpha_max(bad_X, bad_y),
                         lambda: lasso_subgradient(bad_X, bad_y),
                         lambda: kkt_report(bad_X, bad_y, [0]),
                         lambda: dual_report(bad_X, bad_y, [0])]:
                with self.assertRaises(ValueError):
                    call()
        for kw in [dict(alpha=-1), dict(l1_ratio=-.1),
                   dict(l1_ratio=1.1), dict(alpha=True),
                   dict(tol=-1), dict(tol=math.nan),
                   dict(max_iter=-1), dict(max_iter=1.5),
                   dict(max_iter=True), dict(w0=[]), dict(w0=[0, 1]),
                   dict(step=0), dict(step=-1), dict(step=math.inf),
                   dict(fit_intercept=1), dict(split="wrong")]:
            with self.assertRaises(ValueError):
                ista(X, y, **kw)
        for rho in [0, -1, 2, math.nan, True]:
            with self.assertRaises(ValueError):
                alpha_max(X, y, rho)
        for f in [kkt_report, dual_report]:
            for kw in [dict(b=1, fit_intercept=False),
                       dict(b=math.nan), dict(fit_intercept=0),
                       dict(alpha=-1), dict(l1_ratio=2)]:
                with self.assertRaises(ValueError):
                    f(X, y, [0], **kw)
            with self.assertRaises(ValueError):
                f(X, y, [])
        for kw in [dict(decay=-1), dict(step=0), dict(decay=math.inf)]:
            with self.assertRaises(ValueError):
                lasso_subgradient(X, y, **kw)

    def test_sklearn_boundary_comparisons(self):
        # Явный библиотечный мост: solver остаётся чистым Python.
        import numpy as np
        import sklearn
        from sklearn.linear_model import ElasticNet, Lasso
        from sklearn.linear_model import LinearRegression, Ridge
        REPORT["sklearn_version"] = sklearn.__version__
        X, y = data()
        maximum = 0.
        for intercept in [False, True]:
            for alpha, rho in [(0., 1.), (.2, 0.), (.2, .35),
                               (.2, 1.), (10., .5), (10., 1.)]:
                kw = dict(alpha=alpha, l1_ratio=rho,
                          fit_intercept=intercept, tol=1e-10)
                fit = ista(X, y, max_iter=15000, **kw)
                self.assertTrue(fit["converged"])
                if alpha == 0:
                    ref = LinearRegression(fit_intercept=intercept)
                elif rho == 0:
                    ref = Ridge(alpha=len(y) * alpha, solver="svd",
                                fit_intercept=intercept)
                elif rho == 1:
                    ref = Lasso(alpha=alpha, fit_intercept=intercept,
                                tol=1e-12, max_iter=100000)
                else:
                    ref = ElasticNet(alpha=alpha, l1_ratio=rho,
                                     fit_intercept=intercept,
                                     tol=1e-12, max_iter=100000)
                ref.fit(np.array(X), np.array(y))
                delta = max(abs(a - b) for a, b in zip(
                    fit["coef"], ref.coef_))
                maximum = max(maximum, delta)
                self.assertLess(delta, 2e-8)
                self.assertAlmostEqual(fit["intercept"], ref.intercept_,
                                       delta=1e-8)
                report = kkt_report(X, y, fit["coef"], fit["intercept"],
                                    alpha, rho, intercept)
                self.assertLess(report["max_violation"], 2e-10)
                REPORT["comparisons"].append({
                    "alpha": alpha, "rho": rho, "intercept": intercept,
                    "max_coef_error": float(delta),
                    "n_iter": fit["n_iter"],
                })
        REPORT["max_sklearn_coef_error"] = maximum

    def test_duplicate_columns_and_underdetermined(self):
        import numpy as np
        from sklearn.linear_model import ElasticNet, Lasso, Ridge
        X = [[-1., -1., 0.], [1., 1., 0.], [2., 2., 0.],
             [-2., -2., 0.]]
        y = [-2., 2., 4., -4.]
        for rho in [0., .5, 1.]:
            fit = ista(X, y, alpha=.4, l1_ratio=rho,
                       w0=[1., -1., 7.], tol=1e-10)
            cd = coordinate_descent(X, y, alpha=.4, l1_ratio=rho,
                                    tol=1e-10)
            ref = (Lasso(alpha=.4, tol=1e-12) if rho == 1. else
                   Ridge(alpha=len(y) * .4, solver="svd") if rho == 0.
                   else ElasticNet(alpha=.4, l1_ratio=rho, tol=1e-12))
            ref.fit(np.array(X), np.array(y))
            self.assert_vector(predict(X, fit["coef"], fit["intercept"]),
                               ref.predict(X), 2e-8)
            self.assertAlmostEqual(fit["objective"], cd["objective"])
            if rho < 1:
                self.assert_vector(fit["coef"], ref.coef_, 2e-8)
                self.assertAlmostEqual(fit["coef"][0], fit["coef"][1],
                                       delta=1e-8)
        X = [[1., 0., 1., 2.], [0., 1., 1., -1.]]
        y = [1., 3.]
        fit = ista(X, y, alpha=.2, l1_ratio=0, fit_intercept=False,
                   tol=1e-11)
        ref = Ridge(alpha=len(y) * .2, solver="svd", fit_intercept=False)
        ref.fit(np.array(X), np.array(y))
        self.assertTrue(fit["converged"])
        self.assert_vector(fit["coef"], ref.coef_, 1e-9)

    def test_standard_library_only_and_print_width(self):
        path = Path(__file__).with_name("optim_extra.py")
        source = path.read_text()
        self.assertLessEqual(max(map(len, source.splitlines())), 76)
        imports = [node for node in ast.walk(ast.parse(source))
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in imports:
            names = ([node.module] if isinstance(node, ast.ImportFrom)
                     else [alias.name for alias in node.names])
            self.assertTrue(set(names) <= {"math", "pure_core"})


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        OptimizationTests)
    output = io.StringIO()
    with contextlib.redirect_stderr(output):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(output.getvalue())
    REPORT.update({"tests_run": result.testsRun,
                   "successful": result.wasSuccessful(),
                   "failures": len(result.failures),
                   "errors": len(result.errors),
                   "coverage": ["manual KKT and dual", "alpha_max",
                                "ISTA identical splits", "history cost",
                                "zero spectrum", "fixed budgets",
                                "coefficient bound", "invalid inputs",
                                "sklearn endpoints", "duplicate columns",
                                "p>n", "standard library"]})
    print(json.dumps(REPORT, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
