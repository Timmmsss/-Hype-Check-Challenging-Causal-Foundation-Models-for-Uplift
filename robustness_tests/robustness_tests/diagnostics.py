"""Validity diagnostics for the resampled regimes.

Keith et al. (arXiv:2307.15176) §4.4 asks an evaluation designer to check the
precondition that some covariate is dependent on the outcome, and to verify
empirical overlap. ``baseline_benchmark/data.py::_validate_splits`` already
warns below ten positive outcomes per arm; here that rule is promoted from a
``warnings.warn`` to a structured result so a severity can be *flagged* and kept
out of headline tables instead of quietly producing meaningless Qini.

Statistics are numpy-only on purpose: the module stays importable and testable
without scipy. Tail probabilities use the Fisher z-transform with a normal
reference, which is accurate at the sample sizes involved here.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

#: The repo's existing stability rule, from ``data.py::_validate_splits``.
MIN_EVENTS_PER_ARM = 10


def _normal_sf(z: float) -> float:
    """Upper-tail probability of the standard normal."""
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Return a boolean mask of hypotheses rejected at FDR ``alpha``."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be a 1D sequence")
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="mergesort")
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passing = p[order] <= thresholds
    rejected = np.zeros(n, dtype=bool)
    if passing.any():
        cutoff = int(np.flatnonzero(passing).max())
        rejected[order[: cutoff + 1]] = True
    return rejected


def correlation_with_outcome(X, y) -> tuple[np.ndarray, np.ndarray]:
    """Per-column correlation with the outcome, and its two-sided p-value.

    With a binary ``y`` the Pearson correlation is the point-biserial
    correlation. Constant columns get ``r = 0`` and ``p = 1``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    n = len(y)
    if n < 4:
        raise ValueError("At least four rows are needed to test dependence")

    y_centered = y - y.mean()
    y_norm = float(np.sqrt(np.sum(y_centered**2)))
    X_centered = X - X.mean(axis=0, keepdims=True)
    x_norm = np.sqrt(np.sum(X_centered**2, axis=0))

    r = np.zeros(X.shape[1], dtype=float)
    usable = (x_norm > 1e-12) & (y_norm > 1e-12)
    if usable.any():
        r[usable] = (y_centered @ X_centered[:, usable]) / (x_norm[usable] * y_norm)
    r = np.clip(r, -0.999999, 0.999999)

    p = np.ones(X.shape[1], dtype=float)
    if n > 3:
        z = np.arctanh(np.abs(r)) * math.sqrt(n - 3)
        p[usable] = np.array([2.0 * _normal_sf(value) for value in z[usable]])
    return r, np.clip(p, 0.0, 1.0)


def check_outcome_dependent_covariate(
    X,
    y,
    feature_names: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    top_k: int = 5,
    strict: bool = True,
) -> dict[str, Any]:
    """Keith et al.'s precondition: at least one covariate depends on ``Y``.

    Without it there is nothing for a covariate-based rule to grip, and the
    conversion axis cannot move the base rate. ``strict=True`` raises instead of
    reporting, because silently sweeping a severity grid that cannot bite is
    worse than failing.
    """
    r, p = correlation_with_outcome(X, y)
    rejected = benjamini_hochberg(p, alpha=alpha)
    names = list(feature_names) if feature_names is not None else [
        f"feature_{i}" for i in range(len(r))
    ]
    if len(names) != len(r):
        raise ValueError("feature_names must match the number of columns in X")

    ranking = np.argsort(-np.abs(r))[: max(1, int(top_k))]
    result: dict[str, Any] = {
        "n_covariates": int(len(r)),
        "n_dependent": int(rejected.sum()),
        "alpha": float(alpha),
        "max_abs_correlation": float(np.max(np.abs(r))) if len(r) else 0.0,
        "top_covariates": [
            {
                "name": names[int(i)],
                "correlation": float(r[int(i)]),
                "p_value": float(p[int(i)]),
                "dependent": bool(rejected[int(i)]),
            }
            for i in ranking
        ],
        "precondition_met": bool(rejected.any()),
    }
    if strict and not result["precondition_met"]:
        raise ValueError(
            "No covariate is dependent on the outcome at FDR "
            f"{alpha}; the RCT-subsampling precondition of Keith et al. (2023) "
            "is not met for this dataset, so a covariate-based conversion rule "
            "cannot shift the base rate."
        )
    return result


def check_overlap(
    X,
    t,
    seed: int = 0,
    max_rows: int = 20_000,
    n_folds: int = 3,
    clip: float = 0.01,
) -> dict[str, Any]:
    """Empirical overlap: is ``0 < P(T=1|C) < 1`` in practice?

    Cross-fitted so the estimate is not the in-sample optimism of a boosted
    classifier. Reported, not enforced: on an untouched RCT the propensity is
    flat by construction, and it is only the confounded extension where this
    can genuinely fail.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = np.asarray(X, dtype=float)
    t = np.asarray(t).reshape(-1).astype(int)
    if len(X) != len(t):
        raise ValueError("X and t must have the same number of rows")
    if len(np.unique(t)) != 2:
        raise ValueError("check_overlap needs both treatment arms present")

    rng = np.random.default_rng(seed)
    if max_rows and len(X) > max_rows:
        index = np.sort(rng.choice(len(X), size=int(max_rows), replace=False))
        X, t = X[index], t[index]

    smallest_arm = int(min(np.sum(t == 0), np.sum(t == 1)))
    folds = max(2, min(int(n_folds), smallest_arm))
    model = HistGradientBoostingClassifier(
        max_iter=60, learning_rate=0.1, early_stopping=False, random_state=seed
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probability = cross_val_predict(model, X, t, cv=splitter, method="predict_proba")[:, 1]

    outside = float(np.mean((probability <= clip) | (probability >= 1.0 - clip)))
    return {
        "n_scored": int(len(probability)),
        "min_propensity": float(probability.min()),
        "max_propensity": float(probability.max()),
        "mean_propensity": float(probability.mean()),
        "fraction_outside_clip": outside,
        "clip": float(clip),
        "overlap_ok": bool(probability.min() > 0.0 and probability.max() < 1.0),
        "overlap_practical": bool(outside < 0.01),
    }


def check_event_counts(t, y, split_name: str = "split") -> dict[str, Any]:
    """Per-arm positive-outcome counts, with the repo's ``< 10`` rule applied.

    Mirrors ``data.py::_validate_splits`` but returns the verdict rather than
    warning, so the runner can mark a cell degenerate and the analysis can drop
    it from headline tables with a reason attached.
    """
    t = np.asarray(t).reshape(-1).astype(int)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(t) != len(y):
        raise ValueError("t and y must have the same length")

    per_arm: dict[str, dict[str, int]] = {}
    for arm in (0, 1):
        mask = t == arm
        per_arm[f"arm_{arm}"] = {
            "n": int(mask.sum()),
            "events": int(y[mask].sum()) if mask.any() else 0,
        }
    both_arms = bool(per_arm["arm_0"]["n"] > 0 and per_arm["arm_1"]["n"] > 0)
    min_events = min(per_arm["arm_0"]["events"], per_arm["arm_1"]["events"])
    return {
        "split": split_name,
        "n": int(len(t)),
        "treatment_rate": float(np.mean(t)) if len(t) else float("nan"),
        "outcome_rate": float(np.mean(y)) if len(y) else float("nan"),
        **per_arm,
        "min_events_per_arm": int(min_events),
        "both_arms_present": both_arms,
        "degenerate": bool(not both_arms or min_events < MIN_EVENTS_PER_ARM),
        "min_events_threshold": MIN_EVENTS_PER_ARM,
    }


def realized_severity(t, y) -> dict[str, float]:
    """What the resampling actually achieved, as opposed to what was asked for."""
    t = np.asarray(t).reshape(-1).astype(float)
    y = np.asarray(y, dtype=float).reshape(-1)
    return {
        "realized_treatment_rate": float(np.mean(t)) if len(t) else float("nan"),
        "realized_conversion_rate": float(np.mean(y)) if len(y) else float("nan"),
        "realized_control_share": float(np.mean(1.0 - t)) if len(t) else float("nan"),
        "n_rows": int(len(t)),
    }
