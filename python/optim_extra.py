"""Сертификаты и методы первого порядка: стандартная библиотека.

J = ||y-Xw-b||²/(2n) + lambda1*||w||_1 + lambda2*||w||²/2.
Свободный член не штрафуется. При его оценке методы центрируют данные.
Истории включают старт; tol задаёт абсолютную невязку условий KKT.
"""

import math

from pure_core import (
    _dot, _integer, _mean, _number, _objective_residual,
    _penalties, _predict, _total, _training_data, _weights,
    residuals, soft_threshold,
)


def _intercept_flag(fit_intercept):
    if not isinstance(fit_intercept, bool):
        raise ValueError("fit_intercept должен быть bool")


def _center(X, y, fit_intercept):
    _intercept_flag(fit_intercept)
    p = len(X[0])
    means = ([_mean(list(c)) for c in zip(*X)] if fit_intercept
             else [0.0] * p)
    ym = _mean(y) if fit_intercept else 0.0
    columns = [[_number(row[j] - means[j], "Xc") for row in X]
               for j in range(p)]
    yc = [_number(v - ym, "yc") for v in y]
    return list(zip(*columns)), yc, columns, means, ym


def _kkt(columns, r, w, lambda1, lambda2):
    correlations = [_dot(c, r) / len(r) for c in columns]
    violations = []
    for c, v in zip(correlations, w):
        smooth = _number(c - lambda2 * v, "Баланс KKT")
        if v:
            sign = 1 if v > 0 else -1
            error = abs(smooth - lambda1 * sign)
        else:
            error = max(abs(smooth) - lambda1, 0.0)
        violations.append(_number(error, "Невязка KKT"))
    return correlations, violations


def _report_inputs(X, y, w, b, alpha, l1_ratio, fit_intercept):
    X, y = _training_data(X, y)
    w = _weights(w, len(X[0]))
    b = _number(b, "b")
    _intercept_flag(fit_intercept)
    if not fit_intercept and b != 0:
        raise ValueError("При fit_intercept=False требуется b=0")
    lambda1, lambda2 = _penalties(alpha, l1_ratio)
    r = residuals(y, _predict(X, w, b))
    return X, y, w, b, lambda1, lambda2, r


def kkt_report(X, y, w, b=0.0, alpha=0.1, l1_ratio=1.0,
               fit_intercept=True):
    """Невязки KKT; нулевой коэффициент проверяется по интервалу.

    correlations = X^T(y-Xw-b)/n в исходных координатах.
    Для ненулевого w_j: c_j=lambda2*w_j+lambda1*sign(w_j).
    Для w_j=0: |c_j|<=lambda1. При свободном b требуется mean(r)=0.
    """
    args = _report_inputs(X, y, w, b, alpha, l1_ratio,
                          fit_intercept)
    X, y, w, b, lambda1, lambda2, r = args
    correlations, errors = _kkt(list(zip(*X)), r, w,
                                lambda1, lambda2)
    intercept_error = abs(_mean(r)) if fit_intercept else 0.0
    return {
        "max_violation": max(errors + [intercept_error]),
        "coordinate_violations": errors,
        "intercept_violation": intercept_error,
        "correlations": correlations,
    }


def alpha_max(X, y, l1_ratio=1.0, fit_intercept=True):
    """Порог нулевого решения max|Xc^T yc|/(n*rho), rho>0.

    При rho=0 Ridge обычно не имеет конечного порога обнуления.
    Если все корреляции равны нулю, порог равен нулю.
    """
    X, y = _training_data(X, y)
    rho, _ = _penalties(1.0, l1_ratio)
    if rho == 0:
        raise ValueError("Ridge: конечного alpha_max обычно нет (rho=0)")
    _, yc, columns, _, _ = _center(X, y, fit_intercept)
    maximum = max(abs(_dot(c, yc) / len(y)) for c in columns)
    return _number(maximum / rho, "alpha_max")


def _solver_inputs(X, y, alpha, rho, max_iter, tol, w0,
                   fit_intercept):
    X, y = _training_data(X, y)
    lambda1, lambda2 = _penalties(alpha, rho)
    max_iter = _integer(max_iter, "max_iter")
    tol = _number(tol, "tol")
    if tol < 0:
        raise ValueError("tol должен быть неотрицательным")
    Xc, yc, columns, means, ym = _center(X, y, fit_intercept)
    w = [0.0] * len(columns) if w0 is None else _weights(
        w0, len(columns))
    L0 = _total(_mean([v * v for v in c]) for c in columns)
    return (Xc, yc, columns, means, ym, w, lambda1, lambda2,
            max_iter, tol, L0)


def _state(Xc, yc, columns, w, lambda1, lambda2):
    r = residuals(yc, _predict(Xc, w, 0.0))
    correlations, errors = _kkt(columns, r, w, lambda1, lambda2)
    loss = _objective_residual(r, w, lambda1, lambda2)
    return correlations, loss, max(errors)


def ista(X, y, alpha=0.1, l1_ratio=1.0, split="smooth_l2",
         max_iter=10000, tol=1e-8, w0=None, fit_intercept=True,
         step=None):
    """ISTA с L2 в градиенте либо в проксимальном операторе.

    L0=sum_j ||Xc_j||²/n: безопасная, иногда грубая граница спектра.
    Шаги по умолчанию: 1/(L0+lambda2) и 1/L0 соответственно.
    При L0>0 они задают один путь с точностью округления.
    Пользовательский step>0 допускается; сходимость не обещается.
    При L0=0 штраф минимизируется точно одним шагом; без штрафа
    старт уже оптимален. В этом случае step возвращается как None.
    max_iter=0 возвращает проверенный старт без обновлений.
    """
    if split not in ("smooth_l2", "prox_all"):
        raise ValueError("split: нужно 'smooth_l2' или 'prox_all'")
    args = _solver_inputs(X, y, alpha, l1_ratio, max_iter, tol,
                          w0, fit_intercept)
    Xc, yc, cols, means, ym, w, l1, l2, budget, tol, L0 = args
    if step is not None:
        step = _number(step, "step")
        if step <= 0:
            raise ValueError("step должен быть положительным")
    if L0 == 0:
        step = None
    elif step is None:
        denominator = _number(L0 + l2, "L") if split == "smooth_l2" else L0
        step = _number(1 / denominator, "step")
    corr, current, error = _state(Xc, yc, cols, w, l1, l2)
    history, coef_history = [current], [w[:]]
    n_iter = 0
    while n_iter < budget and error > tol:
        n_iter += 1
        if L0 == 0:
            w = [0.0] * len(w)
        elif split == "smooth_l2":
            w = [soft_threshold(v + step * (c - l2 * v), step * l1)
                 for v, c in zip(w, corr)]
        else:
            denominator = _number(1 + step * l2, "Прокс. знаменатель")
            w = [soft_threshold(v + step * c, step * l1) / denominator
                 for v, c in zip(w, corr)]
        corr, current, error = _state(Xc, yc, cols, w, l1, l2)
        history.append(current)
        coef_history.append(w[:])
    converged = error <= tol
    return {
        "coef": w[:], "intercept": _number(ym - _dot(means, w), "b"),
        "objective": current, "history": history,
        "coef_history": coef_history, "n_iter": n_iter,
        "converged": converged,
        "status": "converged" if converged else "max_iter",
        "optimality_error": error, "step": step, "L0": L0,
    }


def lasso_subgradient(X, y, alpha=0.1, max_iter=2000, step=None,
                      decay=0.6, w0=None, fit_intercept=True,
                      tol=1e-8):
    """Фиксированный бюджет: g=-Xc^T r/n+alpha*sign(w), sign(0)=0.

    step_k=step/k**decay; по умолчанию step=1/L0 (1 при L0=0).
    coef — последний шаг, best_coef — лучший посещённый по J.
    История содержит фактические значения J, не накопленный минимум.
    converged проверяет KKT последнего шага; бюджет всегда выполняется.
    status='fixed_budget'; малая разность соседних J не сертификат.
    """
    args = _solver_inputs(X, y, alpha, 1.0, max_iter, tol, w0,
                          fit_intercept)
    Xc, yc, cols, means, ym, w, l1, l2, budget, tol, L0 = args
    decay = _number(decay, "decay")
    if decay < 0:
        raise ValueError("decay должен быть неотрицательным")
    step = (1 / L0 if L0 > 0 else 1.0) if step is None else step
    step = _number(step, "step")
    if step <= 0:
        raise ValueError("step должен быть положительным")
    corr, current, error = _state(Xc, yc, cols, w, l1, l2)
    history, coef_history = [current], [w[:]]
    best_w, best_loss, best_iteration = w[:], current, 0
    for iteration in range(1, budget + 1):
        rate = step * math.exp(-decay * math.log(iteration))
        signs = [(v > 0) - (v < 0) for v in w]
        w = [_number(v + rate * (c - l1 * s), "w")
             for v, c, s in zip(w, corr, signs)]
        corr, current, error = _state(Xc, yc, cols, w, l1, l2)
        history.append(current)
        coef_history.append(w[:])
        if current < best_loss:
            best_w, best_loss, best_iteration = w[:], current, iteration
    return {
        "coef": w[:], "intercept": _number(ym - _dot(means, w), "b"),
        "objective": current, "history": history,
        "coef_history": coef_history, "n_iter": budget,
        "converged": error <= tol, "status": "fixed_budget",
        "optimality_error": error, "step": step, "decay": decay,
        "L0": L0, "best_coef": best_w,
        "best_intercept": _number(ym - _dot(means, best_w), "best_b"),
        "best_objective": best_loss, "best_iteration": best_iteration,
    }


def dual_report(X, y, w, b=0.0, alpha=0.1, l1_ratio=1.0,
                fit_intercept=True):
    """Допустимая двойственная точка и необрезанный зазор J-D.

    theta=r/n; при свободном b проецируем на sum(theta)=0.
    Lasso: ||X^T theta||_inf<=lambda1 обеспечивается масштабированием.
    EN: D=y^T theta-n||theta||²/2-sum S(X^T theta,l1)²/(2*l2).
    При alpha=0 берём theta=0: сертификат верен, но обычно груб.
    coefficient_bound=sqrt(2*gap/lambda2) при lambda2>0, gap>=0;
    это евклидова граница ошибки w, не включающая свободный член.
    Отрицательный gap сохраняется; при нём граница равна None.
    """
    args = _report_inputs(X, y, w, b, alpha, l1_ratio,
                          fit_intercept)
    X, y, w, b, l1, l2, r = args
    n = len(y)
    theta = [v / n for v in r]
    if fit_intercept:
        center = _mean(theta)
        theta = [v - center for v in theta]
        # Удаляем остаток проекции в последней компоненте.
        theta[-1] = -_total(theta[:-1])
    columns = list(zip(*X))
    if l2 > 0:
        method = "elasticnet_conjugate"
    elif l1 > 0:
        method = "lasso_rescaled"
        norm = max(abs(_dot(c, theta)) for c in columns)
        scale = min(1.0, l1 / norm) if norm else 1.0
        theta = [v * scale for v in theta]
    else:
        method = "unpenalized_zero_certificate"
        theta = [0.0] * n
    correlations = [_dot(c, theta) for c in columns]
    conjugate = 0.0
    if l2 > 0:
        shrunk = [soft_threshold(c, l1) for c in correlations]
        conjugate = _total((v / l2) * v / 2 for v in shrunk)
    dual = _total([_dot(y, theta), -n * _dot(theta, theta) / 2,
                   -conjugate])
    primal = _objective_residual(r, w, l1, l2)
    gap = _number(primal - dual, "Двойственный зазор")
    intercept_error = abs(_total(theta)) if fit_intercept else 0.0
    penalty_error = (max(max(abs(c) for c in correlations) - l1, 0.0)
                     if l2 == 0 else 0.0)
    bound = None
    if l2 > 0 and gap >= 0:
        bound = _number(math.sqrt(2) * math.sqrt(gap) / math.sqrt(l2),
                        "Граница ошибки коэффициентов")
    return {
        "primal": primal, "dual": dual, "gap": gap, "theta": theta,
        "feasibility_error": max(intercept_error, penalty_error),
        "method": method, "coefficient_bound": bound,
    }
