"""Semi-synthetic RCTs with a known CATE, for PEHE and the H2 agreement test.

Real campaign data reveal only one potential outcome per unit, so PEHE is not
computable there -- ``BASELINE_FULL_WORKFLOW_EN.md`` §5 says exactly this. The
published semi-synthetic benchmarks (IHDP, ACIC 2016) have continuous outcomes,
which the existing harness rejects: ``metrics.py::_as_arrays`` requires a binary
outcome and ``models.py`` fits classifiers.

So the primary ground-truth source here is a *binary-outcome* generator over
real covariates::

    mu0(x) = sigma(alpha + z(x))         alpha solved for the target base rate
    tau(x) = heterogeneous, known exactly
    y      ~ Bernoulli(clip(mu0 + T * tau))

Because both arm probabilities are clipped before the draw, the true CATE is
``p1(x) - p0(x)`` *exactly*, not approximately. Every existing estimator and
every existing metric works on it unchanged, and it accepts the same axis knobs,
so all three regimes can be swept with ground truth present -- which is what a
per-regime Spearman agreement table requires.

The generator follows the shape already used by
``baseline_benchmark/tests/test_smoke.py::synthetic_rct``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401  (side effect: puts baseline_benchmark on sys.path)
from .prepare import ResampledData, build_resampled_split

from baseline_benchmark.data import (  # noqa: E402
    DATASET_SPECS,
    _make_preprocessor,
    load_cleaned_dataset,
)

PROBABILITY_EPSILON = 1e-3


@dataclass
class SemiSyntheticSpec:
    """The knobs of the generator, recorded verbatim in the run manifest."""

    covariate_source: str = "gaussian:20000x10"
    base_rate: float = 0.05
    treatment_rate: float = 0.5
    ate: float = 0.01
    heterogeneity: float = 1.0
    n_signal_features: int = 5
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return {
            "covariate_source": self.covariate_source,
            "base_rate": self.base_rate,
            "treatment_rate": self.treatment_rate,
            "ate": self.ate,
            "heterogeneity": self.heterogeneity,
            "n_signal_features": self.n_signal_features,
            "seed": self.seed,
        }


# --------------------------------------------------------------------------
# Covariate sources
# --------------------------------------------------------------------------
def _gaussian_covariates(spec: str, rng: np.random.Generator) -> pd.DataFrame:
    body = spec.split(":", 1)[1] if ":" in spec else "20000x10"
    rows, _, columns = body.partition("x")
    n, d = int(rows), int(columns)
    if n < 100 or d < 2:
        raise ValueError("gaussian covariates need at least 100 rows and 2 columns")
    values = rng.normal(size=(n, d))
    return pd.DataFrame(values, columns=[f"x{i}" for i in range(d)])


def load_ihdp_covariates(path: Path, replication: int = 0) -> pd.DataFrame:
    """The 25 IHDP covariates from the standard ``ihdp_npci_*.npz`` files.

    Only the covariates are taken. The response surfaces are supplied by the
    binary generator below, so that the existing binary-outcome estimators and
    metrics apply without a parallel regression code path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"IHDP archive not found at {path}")
    with np.load(path) as archive:
        if "x" not in archive:
            raise ValueError(f"{path} does not look like an IHDP npz file (no 'x' array)")
        x = np.asarray(archive["x"], dtype=float)
    if x.ndim == 3:  # (n, d, replications)
        if not 0 <= replication < x.shape[2]:
            raise ValueError(f"replication must lie in [0, {x.shape[2]})")
        x = x[:, :, replication]
    return pd.DataFrame(x, columns=[f"ihdp_x{i}" for i in range(x.shape[1])])


def load_covariates(
    source: str,
    rng: np.random.Generator,
    cleaned_root: Optional[Path] = None,
    max_rows: Optional[int] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, bool]:
    """Resolve a covariate source to a raw feature frame and its group-safe flag.

    ``gaussian:<n>x<d>``  -- self-contained, needs no data on disk.
    ``ihdp:<path>``       -- the 25 IHDP covariates.
    a cleaned dataset name -- real campaign covariates; the original treatment
    and outcome columns are discarded, only ``X`` is kept.
    """
    key = source.strip().lower()
    if key.startswith("gaussian"):
        return _gaussian_covariates(key, rng), False
    if key.startswith("ihdp"):
        _, _, path = key.partition(":")
        if not path:
            raise ValueError("ihdp source must be given as 'ihdp:<path to npz>'")
        return load_ihdp_covariates(Path(path)), False
    if key in DATASET_SPECS:
        if cleaned_root is None:
            raise ValueError(f"cleaned_root is required to use {key!r} as a covariate source")
        X, _, _, _, _, group_safe = load_cleaned_dataset(
            cleaned_root=cleaned_root,
            dataset=key,
            outcome=None,
            max_rows=max_rows,
            seed=seed,
        )
        return X.reset_index(drop=True), group_safe
    raise ValueError(
        f"Unknown covariate source {source!r}; use 'gaussian:<n>x<d>', 'ihdp:<path>', "
        f"or one of {sorted(DATASET_SPECS)}"
    )


# --------------------------------------------------------------------------
# The generator
# --------------------------------------------------------------------------
def _numeric_matrix(X: pd.DataFrame) -> np.ndarray:
    """Standardised numeric encoding used only to build the response surfaces."""
    preprocessor = _make_preprocessor(X)
    matrix = np.asarray(preprocessor.fit_transform(X), dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Covariate source produced non-finite values")
    return matrix


def _unit_signal(matrix: np.ndarray, rng: np.random.Generator, n_signal: int) -> np.ndarray:
    """A sparse random linear index over the covariates, standardised."""
    d = matrix.shape[1]
    k = int(min(max(1, n_signal), d))
    chosen = rng.choice(d, size=k, replace=False)
    weights = rng.normal(size=k)
    signal = matrix[:, chosen] @ weights
    spread = float(np.std(signal))
    if spread < 1e-12:
        raise ValueError("Selected covariates carry no variation; cannot build a signal")
    return (signal - float(np.mean(signal))) / spread


def _solve_intercept(signal: np.ndarray, base_rate: float) -> float:
    """Bisection for ``alpha`` such that ``mean(sigmoid(alpha + signal)) == base_rate``."""
    if not 0.0 < base_rate < 1.0:
        raise ValueError("base_rate must lie strictly between 0 and 1")

    def mean_probability(alpha: float) -> float:
        return float(np.mean(1.0 / (1.0 + np.exp(-(alpha + signal)))))

    low, high = -50.0, 50.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if mean_probability(middle) < base_rate:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def make_binary_semisynthetic(
    X: pd.DataFrame,
    spec: SemiSyntheticSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Draw ``(t, y, tau_true, info)`` for a binary-outcome semi-synthetic RCT."""
    if not 0.0 < spec.treatment_rate < 1.0:
        raise ValueError("treatment_rate must lie strictly between 0 and 1")

    matrix = _numeric_matrix(X)
    n = len(matrix)
    baseline_signal = _unit_signal(matrix, rng, spec.n_signal_features)
    # A second, differently weighted index so that effect heterogeneity is not a
    # deterministic function of baseline risk -- otherwise ranking by predicted
    # outcome would trivially solve the uplift task.
    effect_signal = _unit_signal(matrix, rng, spec.n_signal_features)

    alpha = _solve_intercept(baseline_signal, spec.base_rate)
    p0 = 1.0 / (1.0 + np.exp(-(alpha + baseline_signal)))
    tau_raw = spec.ate * (1.0 + spec.heterogeneity * np.tanh(effect_signal))
    p1 = p0 + tau_raw

    p0 = np.clip(p0, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    p1 = np.clip(p1, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    # Ground truth is exact after clipping, not an approximation of it.
    tau_true = p1 - p0

    t = rng.binomial(1, spec.treatment_rate, size=n).astype(np.int8)
    probability = np.where(t == 1, p1, p0)
    y = rng.binomial(1, probability).astype(np.int8)

    info = {
        **spec.as_dict(),
        "n_rows": int(n),
        "n_features_numeric": int(matrix.shape[1]),
        "intercept_alpha": float(alpha),
        "realized_base_rate_control": float(np.mean(p0)),
        "realized_ate_true": float(np.mean(tau_true)),
        "tau_true_std": float(np.std(tau_true)),
        "realized_outcome_rate": float(np.mean(y)),
        "realized_treatment_rate": float(np.mean(t)),
    }
    return t, y, tau_true.astype(float), info


def prepare_semisynthetic_data(
    spec: SemiSyntheticSpec,
    axis: str = "scale",
    severity: Any = None,
    seed: int = 42,
    cleaned_root: Optional[Path] = None,
    max_rows: Optional[int] = None,
    dataset_label: Optional[str] = None,
    **kwargs: Any,
) -> ResampledData:
    """One semi-synthetic cell, carrying the true CATE through to every split."""
    generator = np.random.default_rng([spec.seed, seed])
    X, group_safe = load_covariates(
        spec.covariate_source,
        generator,
        cleaned_root=cleaned_root,
        max_rows=max_rows,
        seed=seed,
    )
    t, y, tau_true, info = make_binary_semisynthetic(X, spec, generator)
    ids = np.arange(len(X), dtype=np.int64)

    label = dataset_label or f"semisynth_{spec.covariate_source.split(':')[0]}"
    resampled = build_resampled_split(
        X=X,
        t=t,
        y=y,
        ids=ids,
        dataset=label,
        outcome_name="semisynthetic_conversion",
        group_safe=group_safe,
        axis=axis,
        severity=severity,
        seed=seed,
        extra_arrays={"tau_true": tau_true},
        **kwargs,
    )
    resampled.diagnostics["semisynthetic"] = info
    return resampled
