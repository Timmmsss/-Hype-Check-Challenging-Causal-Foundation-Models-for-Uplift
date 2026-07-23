"""Effect-accuracy metrics and the PEHE/Qini ranking-agreement test (H2).

Hypothesis H2 in ``main.tex``: ordering methods by effect accuracy (PEHE) agrees
only weakly with ordering them by ranking quality (Qini). Goal G2 asks for that
agreement *inside each degraded regime*, so :func:`ranking_agreement` is applied
per (axis, severity) cell rather than once globally.

Statistics are numpy-only. With the handful of methods being compared, the
Spearman p-value is computed from the exact permutation null; the asymptotic
t-approximation is used only when there are too many methods to enumerate.
"""
from __future__ import annotations

import math
from itertools import permutations
from typing import Any, Mapping, Optional, Sequence

import numpy as np

#: Above this many methods, enumerate-the-null becomes impractical.
_EXACT_PERMUTATION_LIMIT = 8


def pehe(tau_true, tau_pred) -> float:
    """Precision in Estimating Heterogeneous Effects: ``sqrt(mean (tau_hat - tau)^2)``."""
    tau_true = np.asarray(tau_true, dtype=float).reshape(-1)
    tau_pred = np.asarray(tau_pred, dtype=float).reshape(-1)
    if tau_true.shape != tau_pred.shape:
        raise ValueError("tau_true and tau_pred must have the same shape")
    if len(tau_true) == 0:
        raise ValueError("tau_true and tau_pred must not be empty")
    if not np.isfinite(tau_true).all() or not np.isfinite(tau_pred).all():
        raise ValueError("tau_true and tau_pred must be finite")
    return float(np.sqrt(np.mean((tau_pred - tau_true) ** 2)))


def abs_ate_error(tau_true, tau_pred) -> float:
    """``|E[tau_hat] - E[tau]|``: the bias part of PEHE, reported alongside it."""
    tau_true = np.asarray(tau_true, dtype=float).reshape(-1)
    tau_pred = np.asarray(tau_pred, dtype=float).reshape(-1)
    if tau_true.shape != tau_pred.shape:
        raise ValueError("tau_true and tau_pred must have the same shape")
    return float(abs(np.mean(tau_pred) - np.mean(tau_true)))


def average_ranks(values) -> np.ndarray:
    """Ranks of ``values``, ascending, ties sharing their average rank."""
    values = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    unique, inverse = np.unique(values, return_inverse=True)
    if len(unique) < len(values):
        sums = np.bincount(inverse, weights=ranks)
        counts = np.bincount(inverse)
        ranks = (sums / counts)[inverse]
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
    if denominator < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def _betacf(a: float, b: float, x: float, iterations: int = 200) -> float:
    """Continued-fraction expansion for the incomplete beta function."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


def _incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_two_sided(t_statistic: float, df: float) -> float:
    if df <= 0:
        return float("nan")
    x = df / (df + t_statistic**2)
    return float(min(1.0, _incomplete_beta(df / 2.0, 0.5, x)))


def spearman(a, b) -> tuple[float, float]:
    """Spearman rho and a two-sided p-value.

    The p-value is exact (enumerating every permutation of the ranks) whenever
    the number of items allows it, which is the normal case here: the items are
    the handful of methods being compared.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Both sequences must have the same length")
    n = len(a)
    if n < 3:
        return float("nan"), float("nan")

    rank_a, rank_b = average_ranks(a), average_ranks(b)
    rho = _pearson(rank_a, rank_b)
    if not np.isfinite(rho):
        return float("nan"), float("nan")

    if n <= _EXACT_PERMUTATION_LIMIT:
        at_least_as_extreme = 0
        total = 0
        for permuted in permutations(range(n)):
            candidate = _pearson(rank_a, rank_b[list(permuted)])
            total += 1
            if np.isfinite(candidate) and abs(candidate) >= abs(rho) - 1e-12:
                at_least_as_extreme += 1
        return float(rho), float(at_least_as_extreme / total)

    if abs(rho) >= 1.0:
        return float(rho), 0.0
    t_statistic = rho * math.sqrt((n - 2) / (1.0 - rho**2))
    return float(rho), _student_t_two_sided(t_statistic, n - 2)


def ranking_agreement(
    pehe_by_model: Mapping[str, float],
    qini_by_model: Mapping[str, float],
    models: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Spearman agreement between the PEHE ranking and the Qini ranking (H2).

    Both are converted to *quality* ranks first -- rank 1 is the best method on
    that criterion, meaning lowest PEHE or highest Qini -- so ``rho > 0`` means
    the two criteria agree and ``rho`` near zero is the H2 prediction. The raw
    value-level correlation is reported too, since it is the quantity a reader
    may expect to be negative under agreement.
    """
    names = list(models) if models is not None else sorted(
        set(pehe_by_model) & set(qini_by_model)
    )
    usable = [
        name
        for name in names
        if name in pehe_by_model
        and name in qini_by_model
        and np.isfinite(pehe_by_model[name])
        and np.isfinite(qini_by_model[name])
    ]
    if len(usable) < 3:
        return {
            "n_methods": len(usable),
            "spearman_rho": float("nan"),
            "p_value": float("nan"),
            "value_correlation": float("nan"),
            "models": usable,
            "note": "at least three methods with both metrics are required",
        }

    pehe_values = np.array([pehe_by_model[name] for name in usable], dtype=float)
    qini_values = np.array([qini_by_model[name] for name in usable], dtype=float)
    # Quality ranks: ascending PEHE is already best-first; Qini must be flipped.
    pehe_quality = average_ranks(pehe_values)
    qini_quality = average_ranks(-qini_values)
    rho, p_value = spearman(pehe_quality, qini_quality)
    return {
        "n_methods": len(usable),
        "spearman_rho": rho,
        "p_value": p_value,
        "value_correlation": _pearson(pehe_values, qini_values),
        "models": usable,
        "pehe_order": [usable[i] for i in np.argsort(pehe_values)],
        "qini_order": [usable[i] for i in np.argsort(-qini_values)],
    }
