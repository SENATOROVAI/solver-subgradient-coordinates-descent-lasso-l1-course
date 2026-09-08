"""Мост к NumPy/sklearn: source audit, CV и сравнение ветвей."""
import hashlib
import inspect
from pathlib import Path
import warnings
import numpy as np
import sklearn
from scipy.sparse import csc_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNet, Lasso, MultiTaskElasticNet, enet_path, _cd_fast)
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SNAPSHOT = "cc50648cc1b759b53a4edbce0f3bb6c237349448"
RELEASE = "646da0f072a8afef6a980aa427a710311e67eb9d"


def source_identity():
    """Сверить доступные source-файлы; это не пересборка .so."""
    root = Path(sklearn.__file__).parent
    expected = {
        "linear_model/_cd_fast.pyx":
            "c227100ec066e71ea4b122efc203989e65aec4d1",
        "linear_model/_coordinate_descent.py":
            "efa5a76adfad5e4f0dcf196c8a4de433e702b4cf",
        "_loss/loss.py":
            "9cbaa5284d3a207d5905049a73c5db2704d2aff0",
    }
    hashes = {}
    for name, target in expected.items():
        path = root / name
        if path.exists():
            data = path.read_bytes()
            prefix = b"blob " + str(len(data)).encode() + b"\0"
            value = hashlib.sha1(prefix + data).hexdigest()
            hashes[name] = {"sha": value, "matches_1_8": value == target}
        else:
            hashes[name] = {"status": "source_not_distributed"}
    fit = inspect.unwrap(ElasticNet.fit)
    return {"version": sklearn.__version__, "release": RELEASE,
            "snapshot": SNAPSHOT, "hashes": hashes,
            "inherited_fit": Lasso.fit is ElasticNet.fit,
            "path_is_enet": Lasso.path is enet_path,
            "fit_source": inspect.getsourcefile(fit),
            "fit_start": inspect.getsourcelines(fit)[1]}


def make_model(alpha, rho=.7):
    return make_pipeline(StandardScaler(), ElasticNet(
        alpha=float(alpha), l1_ratio=rho,
        max_iter=30000, tol=1e-9))


def cv_one_se(X, y, alphas, rho=.7, folds=5, seed=620):
    """Largest alpha within SE at minimum; fixed rho, same folds."""
    X, y = np.asarray(X), np.asarray(y)
    split = list(KFold(folds, shuffle=True,
                      random_state=seed).split(X))
    scores, scaler_counts = [], []
    for candidate, alpha in enumerate(alphas):
        row = []
        for train, valid in split:
            model = make_model(alpha, rho)
            model.fit(X[train], y[train])
            if candidate == 0:
                scaler_counts.append(int(model.named_steps[
                    "standardscaler"].n_samples_seen_))
            row.append(np.mean((y[valid] - model.predict(X[valid])) ** 2))
        scores.append(row)
    scores = np.asarray(scores)
    mean = scores.mean(axis=1)
    se = scores.std(axis=1, ddof=1) / np.sqrt(folds)
    best = int(np.argmin(mean))
    limit = mean[best] + se[best]
    choices = np.flatnonzero(mean <= limit)
    one_se = int(max(choices, key=lambda i: alphas[i]))
    return {"fold_scores": scores.tolist(), "mean": mean.tolist(),
            "se": se.tolist(), "best_index": best,
            "one_se_index": one_se, "threshold": float(limit),
            "best_alpha": float(alphas[best]),
            "one_se_alpha": float(alphas[one_se]),
            "scaler_training_counts": scaler_counts}


def dual_pair(X, y, w, lambda1=.2, lambda2=.8):
    """Два корректных сертификата; dense, lambda1,lambda2 > 0."""
    X, y, w = np.asarray(X), np.asarray(y), np.asarray(w)
    if lambda1 <= 0 or lambda2 <= 0:
        raise ValueError("Этот пример требует оба штрафа > 0")
    n = len(y)
    r = y - X @ w
    correlation = X.T @ r / n - lambda2 * w
    norm = np.max(np.abs(correlation))
    scale = min(1., lambda1 / norm) if norm > 0 else 1.
    primal = r @ r / (2 * n) + lambda1 * np.abs(w).sum()
    primal += lambda2 / 2 * (w @ w)
    A = scale * (y @ r) / n
    A -= scale ** 2 / 2 * (r @ r / n + lambda2 * (w @ w))
    u = r / n
    excess = np.maximum(np.abs(X.T @ u) - lambda1, 0.)
    B = y @ u - n / 2 * (u @ u)
    B -= excess @ excess / (2 * lambda2)
    return {"primal": float(primal), "D_A": float(A),
            "D_B": float(B), "gap_A": float(primal - A),
            "gap_B": float(primal - B)}


def nested_one_se(X, y, alphas, rho=.7, seed=620):
    """Внешние фолды оценивают заново повторённый внутренний выбор."""
    X, y = np.asarray(X), np.asarray(y)
    records = []
    outer = KFold(3, shuffle=True, random_state=seed)
    for index, (train, valid) in enumerate(outer.split(X)):
        chosen = cv_one_se(X[train], y[train], alphas, rho,
                           folds=4, seed=seed + index + 1)
        model = make_model(chosen["one_se_alpha"], rho)
        model.fit(X[train], y[train])
        error = np.mean((y[valid] - model.predict(X[valid])) ** 2)
        records.append({"alpha": chosen["one_se_alpha"],
                        "mse": float(error)})
    return records


def branch_comparison(X, y, alpha=.2, rho=.6):
    """Плотная задача без intercept; остальные ветви сверяем с ней."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    options = dict(alpha=alpha, l1_ratio=rho, fit_intercept=False,
                   tol=1e-11, max_iter=30000)
    reference = ElasticNet(**options).fit(X, y)
    errors = {}
    variants = {
        "C order": (np.array(X, order="C"), False),
        "F order": (np.array(X, order="F"), False),
        "Gram": (X, X.T @ X),
        "CSC": (csc_matrix(X), False),
    }
    for name, (data, gram) in variants.items():
        model = ElasticNet(precompute=gram, **options).fit(data, y)
        errors[name] = float(np.max(np.abs(
            model.predict(data) - reference.predict(X))))
    raw = _cd_fast.enet_coordinate_descent(
        np.zeros(X.shape[1]), len(y) * alpha * rho,
        len(y) * alpha * (1 - rho), np.asfortranarray(X), y,
        30000, 1e-11, np.random.RandomState(621))
    errors["Cython"] = float(np.max(np.abs(
        X @ raw[0] - reference.predict(X))))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        limited = ElasticNet(**dict(options, max_iter=1)).fit(X, y)
    warning = any(issubclass(w.category, ConvergenceWarning)
                  for w in caught)
    positive = ElasticNet(positive=True, **options).fit(X, y)
    Y = np.column_stack([y, .3 * y + X[:, -1]])
    independent = ElasticNet(**options).fit(X, Y)
    scalar = np.array([ElasticNet(**options).fit(X, t).coef_
                       for t in Y.T])
    joint = MultiTaskElasticNet(**options).fit(X, Y)
    _, path, _ = enet_path(
        X, Y, alphas=[alpha], l1_ratio=rho,
        precompute=False, tol=1e-11, max_iter=30000)
    assert np.max(np.abs(independent.coef_ - scalar)) < 1e-7
    assert np.max(np.abs(joint.coef_ - path[:, :, 0])) < 1e-7
    return {"branch_errors": errors, "coef": reference.coef_.tolist(),
            "positive_coef": positive.coef_.tolist(),
            "gap": float(reference.dual_gap_),
            "private_gap_scaled": float(raw[1] / len(y)),
            "limited_warning": warning,
            "limited_gap": float(limited.dual_gap_),
            "threshold": float(1e-11 * (y @ y) / len(y)),
            "independent": independent.coef_.tolist(),
            "joint": joint.coef_.tolist()}
