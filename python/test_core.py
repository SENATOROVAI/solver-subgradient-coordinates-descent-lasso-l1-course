"""Запуск: python code/test_core.py; проверки не нужны для pure_core."""

import ast
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import sklearn
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import pure_core as pc


REPORT = {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
          "numpy": np.__version__, "comparisons": {}, "manual": {}}


def largest_difference(first, second):
    return float(np.max(np.abs(np.asarray(first) - np.asarray(second))))


class CoreTests(unittest.TestCase):
    def assert_close(self, actual, expected, tolerance=1e-10):
        error = largest_difference(actual, expected)
        self.assertLessEqual(error, tolerance)

    def test_manual_predictions_and_metrics(self):
        X, y = [[1], [2], [3]], [3, 5, 8]
        predictions = pc.predict(X, [2], 1)
        self.assert_close(predictions, [3, 5, 7])
        self.assert_close(pc.residuals(y, predictions), [0, 0, 1])
        self.assertAlmostEqual(pc.mae(y, predictions), 1 / 3)
        self.assertAlmostEqual(pc.mse(y, predictions), 1 / 3)
        self.assertAlmostEqual(pc.mae(y, predictions),
                               mean_absolute_error(y, predictions))
        self.assertAlmostEqual(pc.mse(y, predictions),
                               mean_squared_error(y, predictions))
        REPORT["manual"]["predictions"] = predictions
        REPORT["manual"]["mae_and_mse"] = pc.mae(y, predictions)

    def test_medians_and_grid(self):
        self.assertAlmostEqual(pc.median([9, 1, 4]), 4)
        self.assertAlmostEqual(pc.median([9, 1, 4, 6]), 5)
        self.assertAlmostEqual(pc.median([7]), 7)
        X, y = [[1], [2], [3]], [3, 5, 8]
        result = pc.grid_mae(X, y, [0, 2], [0, 1])
        self.assert_close(result["coef"], [2])
        self.assertAlmostEqual(result["intercept"], 1)
        self.assertAlmostEqual(result["loss"], 1 / 3)
        self.assertEqual(len(result["records"]), 4)
        even_y = [1, 4, 6, 9]
        for b in (4, 5, 6):
            self.assertAlmostEqual(pc.mae(even_y, [b] * 4), 2.5)

    def test_mae_step_and_actual_history(self):
        one = pc.fit_mae([[-1], [1]], [-1, 1], steps=1,
                         step=0.1, w0=[0], b0=0)
        self.assert_close(one["coef"], [0.1])
        self.assertAlmostEqual(one["intercept"], 0)
        self.assert_close(one["history"], [1, 0.9])
        self.assertFalse(one["converged"])
        X, y = [[1], [2], [3]], [3, 5, 8]
        result = pc.fit_mae(X, y, steps=12, step=5, decay=0)
        actual = [pc.mae(y, pc.predict(X, state[:-1], state[-1]))
                  for state in result["parameter_history"]]
        self.assert_close(actual, result["history"])
        self.assertAlmostEqual(result["best_loss"], min(actual))
        self.assertAlmostEqual(result["best_loss"], pc.mae(
            y, pc.predict(X, result["coef"], result["intercept"])))
        self.assertTrue(any(b > a for a, b in zip(actual, actual[1:])))
        self.assertEqual(result["n_iter"], 12)
        self.assertEqual(result["status"], "max_steps")
        zero = pc.fit_mae(X, y, steps=0)
        self.assertEqual(zero["n_iter"], 0)
        self.assertEqual(len(zero["history"]), 1)
        REPORT["manual"]["mae_one_step"] = one

    def test_scaling_and_forecast_equivalence(self):
        X = [[1, 10, 0.7], [3, 20, 0.7], [5, 30, 0.7]]
        stats = pc.scale_fit(X)
        self.assert_close(stats["mean"], [3, 20, 0.7])
        self.assert_close(stats["scale"],
                          [math.sqrt(8 / 3), math.sqrt(200 / 3), 1])
        Z = pc.scale_transform(X, stats)
        self.assert_close(Z, StandardScaler().fit_transform(X))
        w, b = pc.unscale_coefficients([2, -3, 8], 7, stats)
        self.assert_close(pc.predict(Z, [2, -3, 8], 7),
                          pc.predict(X, w, b))
        new_X = [[2, 25, 0.7], [8, 40, 0.7]]
        self.assert_close(pc.predict(pc.scale_transform(new_X, stats),
                                     [2, -3, 8], 7),
                          pc.predict(new_X, w, b))
        self.assert_close(pc.scale_fit([[2]])["scale"], [1])
        REPORT["manual"]["scaling"] = stats

    def test_soft_threshold_and_scalar_minima(self):
        self.assert_close([pc.soft_threshold(z, 0.5)
                           for z in [-2, -0.5, 0, 0.5, 2]],
                          [-1.5, 0, 0, 0, 1.5])
        # X=[-1,1], y=[-2,2]: z=2, q=1.
        X, y = [[-1], [1]], [-2, 2]
        for alpha, rho, expected in [(0.5, 1, 1.5),
                                      (3, 1, 0), (1, 0.5, 1)]:
            fitted = pc.coordinate_descent(X, y, alpha, rho)
            self.assert_close(fitted["coef"], [expected])
            self.assertTrue(fitted["converged"])
            target = (2 - expected) ** 2 / 2
            target += alpha * rho * abs(expected)
            target += alpha * (1 - rho) * expected ** 2 / 2
            self.assertAlmostEqual(fitted["objective"], target)

    def test_hand_orthonormal_geometry(self):
        X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        y = [2 * row[0] + 0.5 * row[1] for row in X]
        lasso = pc.lasso_cd(X, y, alpha=0.8)
        elastic = pc.elasticnet_cd(X, y, alpha=1, l1_ratio=0.5)
        ridge = pc.coordinate_descent(X, y, alpha=0.5, l1_ratio=0)
        self.assert_close(lasso["coef"], [1.2, 0])
        self.assert_close(elastic["coef"], [1, 0])
        self.assert_close(ridge["coef"], [4 / 3, 1 / 3])
        REPORT["manual"]["orthonormal"] = {
            "lasso": lasso["coef"], "elasticnet": elastic["coef"],
            "ridge": ridge["coef"],
        }

    def test_duplicate_handcase_and_nonuniqueness(self):
        X, y = [[-1, -1], [1, 1]], [-3, 3]
        first = pc.lasso_cd(X, y, alpha=0.2, tol=1e-12)
        second = pc.lasso_cd(X, y, alpha=0.2, w0=[0, 2.8],
                             tol=1e-12)
        elastic = pc.elasticnet_cd(X, y, alpha=1, l1_ratio=0.2,
                                   tol=1e-12)
        self.assertAlmostEqual(sum(first["coef"]), 2.8)
        self.assertAlmostEqual(sum(second["coef"]), 2.8)
        self.assertGreater(largest_difference(first["coef"],
                                              second["coef"]), 2)
        self.assert_close(pc.predict(X, first["coef"]),
                          pc.predict(X, second["coef"]))
        self.assertAlmostEqual(first["objective"], 0.58)
        self.assert_close(elastic["coef"], [1, 1])
        self.assertAlmostEqual(elastic["objective"], 1.7)
        REPORT["manual"]["duplicates"] = {
            "lasso_first": first["coef"], "lasso_second": second["coef"],
            "lasso_objective": first["objective"],
            "elasticnet": elastic["coef"],
            "elasticnet_objective": elastic["objective"],
        }

    def test_sklearn_agreement_and_endpoints(self):
        X, y, _ = pc.marketing_data(n=100, p=8, seed=42)
        Z = pc.scale_transform(X, pc.scale_fit(X))
        cases = [
            ("lasso", 0.2, 1.0,
             Lasso(alpha=0.2, tol=1e-13, max_iter=100000)),
            ("elasticnet", 0.3, 0.4,
             ElasticNet(alpha=0.3, l1_ratio=0.4, tol=1e-13,
                        max_iter=100000)),
            ("ridge_endpoint", 0.37, 0,
             Ridge(alpha=len(y) * 0.37, solver="svd")),
            ("ols_endpoint", 0, 0.5, LinearRegression()),
        ]
        for name, alpha, rho, model in cases:
            with self.subTest(name=name):
                pure = pc.coordinate_descent(Z, y, alpha, rho,
                                             tol=1e-11, max_iter=100000)
                model.fit(Z, y)
                predictions = pc.predict(Z, pure["coef"],
                                         pure["intercept"])
                coef_error = largest_difference(pure["coef"], model.coef_)
                pred_error = largest_difference(
                    predictions, model.predict(Z))
                self.assertLessEqual(coef_error, 1e-8)
                self.assertLessEqual(pred_error, 1e-8)
                self.assertAlmostEqual(pure["intercept"], model.intercept_)
                self.assertTrue(pure["converged"])
                self.assertLessEqual(pure["optimality_error"], 1e-11)
                self.assertTrue(all(b <= a + 1e-12 for a, b in zip(
                    pure["history"], pure["history"][1:])))
                REPORT["comparisons"][name] = {
                    "max_coefficient_error": coef_error,
                    "max_prediction_error": pred_error,
                    "n_iter": pure["n_iter"],
                    "optimality_error": pure["optimality_error"],
                }
        lasso = pc.lasso_cd(Z, y, alpha=0.2)
        endpoint = pc.elasticnet_cd(Z, y, alpha=0.2, l1_ratio=1)
        self.assert_close(lasso["coef"], endpoint["coef"])

    def test_no_intercept_and_rank_deficient_ols(self):
        X = [[1, 2], [2, -1], [3, 4], [-1, 1]]
        y = [4, -1, 5, 3]
        pure = pc.elasticnet_cd(X, y, alpha=0.2, l1_ratio=0.4,
                                fit_intercept=False, tol=1e-12)
        reference = ElasticNet(alpha=0.2, l1_ratio=0.4,
                               fit_intercept=False, tol=1e-13,
                               max_iter=100000).fit(X, y)
        self.assertAlmostEqual(pure["intercept"], 0)
        self.assert_close(pure["coef"], reference.coef_)
        duplicate_X = [[-1, -1], [0, 0], [1, 1]]
        duplicate_y = [1, 4, 7]
        ols = pc.coordinate_descent(duplicate_X, duplicate_y, alpha=0)
        ref = LinearRegression().fit(duplicate_X, duplicate_y)
        self.assert_close(pc.predict(duplicate_X, ols["coef"],
                                     ols["intercept"]),
                          ref.predict(duplicate_X))

    def test_trace_objectives_and_iteration_budget(self):
        X, y, _ = pc.marketing_data(n=40, seed=13)
        X = pc.scale_transform(X, pc.scale_fit(X))
        fitted = pc.elasticnet_cd(X, y, alpha=0.2, l1_ratio=0.7,
                                  keep_steps=True, tol=1e-10)
        p = len(X[0])
        self.assertEqual(len(fitted["trace"]), 1 + p * fitted["n_iter"])
        self.assertEqual(len(fitted["history"]), fitted["n_iter"] + 1)
        self.assertEqual(len(fitted["coef_history"]),
                         fitted["n_iter"] + 1)
        means = np.mean(X, axis=0)
        for state in fitted["trace"]:
            b = float(np.mean(y) - np.dot(means, state["coef"]))
            exact = pc.objective(X, y, state["coef"], b, 0.2, 0.7)
            self.assertLessEqual(abs(exact - state["objective"]), 1e-10)
        short = pc.lasso_cd(X, y, max_iter=1, tol=1e-14)
        self.assertEqual(short["n_iter"], 1)
        self.assertEqual(short["status"], "max_iter")
        self.assertFalse(short["converged"])
        self.assertGreater(short["optimality_error"], 1e-14)
        REPORT["comparisons"]["one_epoch_budget"] = {
            "status": short["status"], "n_iter": short["n_iter"],
            "optimality_error": short["optimality_error"],
        }

    def test_zero_constant_all_active_and_all_zero(self):
        for alpha, rho in [(0, 1), (0.2, 1), (0.4, 0.5), (0.2, 0)]:
            result = pc.coordinate_descent([[0, 0.7], [0, 0.7]],
                                           [6, 8], alpha, rho, w0=[5, 8])
            self.assert_close(result["coef"], [0, 0])
            self.assertAlmostEqual(result["intercept"], 7)
            self.assertAlmostEqual(result["objective"], 0.5)
            self.assertTrue(result["converged"])
            self.assertEqual(result["n_iter"], 0)
        X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        y = [2 * row[0] + 3 * row[1] for row in X]
        active = pc.lasso_cd(X, y, alpha=0.1)
        inactive = pc.lasso_cd(X, y, alpha=10)
        self.assertTrue(all(abs(w) > 1 for w in active["coef"]))
        self.assert_close(inactive["coef"], [0, 0])
        self.assertTrue(inactive["converged"])
        single = pc.lasso_cd([[9, 0]], [5])
        self.assert_close(single["coef"], [0, 0])
        self.assertAlmostEqual(single["intercept"], 5)

    def test_generator_exact_order_and_reproducibility(self):
        import random
        rng = random.Random(42)
        z = [rng.gauss(0, 1) for _ in range(5)]
        row = [10 + 2 * z[0], 5 + z[1], 3 + 1.5 * z[2], z[3], 0]
        row[4] = 0.95 * z[0] + math.sqrt(1 - 0.95 ** 2) * rng.gauss(0, 1)
        target = 20 + 3 * row[0] - 2 * row[1] + row[2] + rng.gauss(0, 1)
        X, y, beta = pc.marketing_data(n=1, p=5)
        self.assert_close(X, [row])
        self.assert_close(y, [target])
        self.assert_close(beta, [3, -2, 1, 0, 0])
        for correlation in [-1, 0, 1]:
            small_X, small_y, small_beta = pc.marketing_data(
                n=3, p=3, seed=-42, noise=0, correlation=correlation)
            self.assert_close(pc.predict(small_X, small_beta, 20), small_y)
        first = pc.marketing_data()
        second = pc.marketing_data()
        self.assert_close(first[0], second[0])
        self.assert_close(first[1], second[1])

    def test_invalid_inputs_raise_value_error(self):
        bad_calls = [
            lambda: pc.predict([], [1]),
            lambda: pc.predict([[]], []),
            lambda: pc.predict([[1], [2, 3]], [1]),
            lambda: pc.predict([[1]], [1, 2]),
            lambda: pc.predict([[True]], [1]),
            lambda: pc.predict([[math.nan]], [1]),
            lambda: pc.predict([[1]], [math.inf]),
            lambda: pc.predict([[1]], [1], "0"),
            lambda: pc.predict([[1e308]], [1e308]),
            lambda: pc.residuals([1e308], [-1e308]),
            lambda: pc.mae([], []),
            lambda: pc.mse([1], [1, 2]),
            lambda: pc.median("123"),
            lambda: pc.grid_mae([[1, 2]], [1], [0], [0]),
            lambda: pc.grid_mae([[1]], [1], [], [0]),
            lambda: pc.fit_mae([[1]], [1], steps=1.5),
            lambda: pc.fit_mae([[1]], [1], steps=True),
            lambda: pc.fit_mae([[1]], [1], steps=-1),
            lambda: pc.fit_mae([[1]], [1], step=0),
            lambda: pc.fit_mae([[1]], [1], decay=-0.1),
            lambda: pc.fit_mae([[1]], [1], w0=[0, 0]),
            lambda: pc.scale_transform([[1]], {"mean": [0]}),
            lambda: pc.scale_transform([[1]],
                                       {"mean": [0], "scale": [0]}),
            lambda: pc.unscale_coefficients([1], 0,
                                            {"mean": [0], "scale": [-1]}),
            lambda: pc.soft_threshold(1, -1),
            lambda: pc.soft_threshold(True, 1),
            lambda: pc.coordinate_descent([[1]], [1, 2]),
            lambda: pc.coordinate_descent([[1]], [1], alpha=-1),
            lambda: pc.coordinate_descent([[1]], [1], l1_ratio=1.1),
            lambda: pc.coordinate_descent([[1]], [1], tol=math.inf),
            lambda: pc.coordinate_descent([[1]], [1], tol=-1),
            lambda: pc.coordinate_descent([[1]], [1], max_iter=0),
            lambda: pc.coordinate_descent([[1]], [1], max_iter=1.5),
            lambda: pc.coordinate_descent([[1]], [1], fit_intercept=1),
            lambda: pc.coordinate_descent([[1]], [1], keep_steps="yes"),
            lambda: pc.coordinate_descent([[1]], [1], w0=[math.nan]),
            lambda: pc.objective([[1]], [1], [1], alpha=math.nan),
            lambda: pc.marketing_data(n=0),
            lambda: pc.marketing_data(p=2),
            lambda: pc.marketing_data(noise=-1),
            lambda: pc.marketing_data(correlation=1.1),
            lambda: pc.marketing_data(seed=1.5),
        ]
        for index, call in enumerate(bad_calls):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    call()
        REPORT["invalid_input_cases"] = len(bad_calls)

    def test_core_imports_are_standard_library(self):
        source = Path(pc.__file__).read_text()
        parsed = ast.parse(source)
        imports = []
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0]
                               for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module.split(".")[0])
        self.assertTrue(set(imports) <= {"math", "numbers", "random"})
        REPORT["core_imports"] = imports


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    REPORT["tests_run"] = result.testsRun
    REPORT["failures"] = len(result.failures)
    REPORT["errors"] = len(result.errors)
    REPORT["success"] = result.wasSuccessful()
    destination = Path(__file__).with_name("core_verification.json")
    destination.write_text(json.dumps(REPORT, ensure_ascii=False, indent=2)
                           + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
