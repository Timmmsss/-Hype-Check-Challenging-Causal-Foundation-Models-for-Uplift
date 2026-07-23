"""Test-set bootstrap confidence intervals over saved predictions.

``run_suite.py`` summarises *across seeds* with a normal approximation, which is
not a bootstrap of the evaluation set itself. This module resamples the
evaluation rows with replacement and recomputes the metric, so the interval
reflects the sampling noise of the test split -- the dominant source of Qini
variance at the low event counts these datasets have.

Nothing here retrains a model: it reads the ``predictions.parquet`` files the
existing runners already write, so it also applies retroactively to the
completed part-1/2 runs.

Two design points matter for the H3 claim:

* Within one cell every model is resampled on the **same** row indices, so the
  difference ``CausalPFN - baseline`` gets a properly paired interval. The
  difference, not the levels, is what H3 is about.
* Across seeds the per-seed draws are pooled, giving an interval that carries
  both row noise and split noise.
"""
from __future__ import annotations

import zlib
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401  (side effect: puts baseline_benchmark on sys.path)

from baseline_benchmark.metrics import _area, qini_curve, uplift_at_k  # noqa: E402

DEFAULT_METRICS = ("qini_auc_normalized", "uplift_at_10pct")
CELL_KEYS = ("dataset", "outcome", "axis", "severity_tag", "seed")


def _perfect_qini_area(y: np.ndarray, t: np.ndarray) -> float:
    """Area above random for the oracle ranking; depends only on ``y`` and ``t``.

    Hoisting this out of the per-model loop halves the bootstrap cost, and it
    reproduces ``metrics.qini_auc_normalized`` exactly because it reuses the
    same curve and area helpers.
    """
    perfect_score = y * t - y * (1 - t)
    x, perfect = qini_curve(y, perfect_score, t)
    return _area(x, perfect) - _area(x[[0, -1]], np.array([0.0, perfect[-1]]))


def _qini_auc_normalized(y: np.ndarray, score: np.ndarray, t: np.ndarray, perfect_above: float) -> float:
    x, actual = qini_curve(y, score, t)
    actual_above = _area(x, actual) - _area(x[[0, -1]], np.array([0.0, actual[-1]]))
    if abs(perfect_above) <= 1e-12:
        return float("nan")
    return float(actual_above / perfect_above)


def percentile_interval(draws, level: float = 0.95) -> dict[str, float]:
    """Percentile bootstrap interval, ignoring failed resamples."""
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "n_draws": 0}
    tail = (1.0 - level) / 2.0
    return {
        "ci_low": float(np.quantile(values, tail)),
        "ci_high": float(np.quantile(values, 1.0 - tail)),
        "n_draws": int(len(values)),
    }


def _wide_predictions(cell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Align every model's scores onto one shared row order."""
    required = {"epk_id", "model", "T", "Y", "cate_pred"}
    missing = required - set(cell.columns)
    if missing:
        raise ValueError(f"predictions are missing column(s) {sorted(missing)}")

    models = sorted(cell["model"].astype(str).unique())
    reference = cell.loc[cell["model"].astype(str) == models[0]].sort_values("epk_id")
    ids = reference["epk_id"].to_numpy()
    y = reference["Y"].to_numpy(dtype=float)
    t = reference["T"].to_numpy(dtype=float)

    scores: dict[str, np.ndarray] = {}
    for model in models:
        frame = cell.loc[cell["model"].astype(str) == model].sort_values("epk_id")
        if len(frame) != len(reference) or not np.array_equal(frame["epk_id"].to_numpy(), ids):
            raise ValueError(
                f"model {model!r} was evaluated on a different row set than {models[0]!r}; "
                "a paired bootstrap needs identical evaluation rows"
            )
        scores[model] = frame["cate_pred"].to_numpy(dtype=float)
    return y, t, scores


def bootstrap_cell(
    cell: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 0,
    metrics: Sequence[str] = DEFAULT_METRICS,
    max_rows: Optional[int] = None,
    level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap every model in one (dataset, axis, severity, seed) cell.

    Returns per-model draws keyed by metric, so the caller can pool them across
    seeds and form paired differences without re-running anything.
    """
    y, t, scores = _wide_predictions(cell)
    n = len(y)
    if n < 2:
        raise ValueError("A bootstrap needs at least two evaluation rows")
    unknown = set(metrics) - {"qini_auc_normalized", "uplift_at_10pct", "uplift_at_20pct"}
    if unknown:
        raise ValueError(f"Unsupported bootstrap metric(s) {sorted(unknown)}")

    rng = np.random.default_rng(seed)
    if max_rows and n > int(max_rows):
        subset = np.sort(rng.choice(n, size=int(max_rows), replace=False))
        y, t = y[subset], t[subset]
        scores = {model: values[subset] for model, values in scores.items()}
        n = len(y)

    point: dict[str, dict[str, float]] = {metric: {} for metric in metrics}
    perfect_above = _perfect_qini_area(y, t)
    for model, values in scores.items():
        for metric in metrics:
            point[metric][model] = _evaluate(metric, y, values, t, perfect_above)

    draws: dict[str, dict[str, np.ndarray]] = {
        metric: {model: np.full(n_bootstrap, np.nan) for model in scores} for metric in metrics
    }
    failures = 0
    for b in range(int(n_bootstrap)):
        index = rng.integers(0, n, size=n)
        y_b, t_b = y[index], t[index]
        # A resample can lose a treatment arm entirely; that draw is dropped
        # rather than silently biasing the interval.
        if t_b.min() == t_b.max():
            failures += 1
            continue
        try:
            perfect_b = _perfect_qini_area(y_b, t_b)
        except ValueError:
            failures += 1
            continue
        for model, values in scores.items():
            score_b = values[index]
            for metric in metrics:
                try:
                    draws[metric][model][b] = _evaluate(metric, y_b, score_b, t_b, perfect_b)
                except ValueError:
                    draws[metric][model][b] = np.nan

    return {
        "n_rows": int(n),
        "n_bootstrap": int(n_bootstrap),
        "n_failed_resamples": int(failures),
        "level": float(level),
        "models": sorted(scores),
        "point": point,
        "draws": draws,
    }


def _evaluate(
    metric: str,
    y: np.ndarray,
    score: np.ndarray,
    t: np.ndarray,
    perfect_above: float,
) -> float:
    if metric == "qini_auc_normalized":
        return _qini_auc_normalized(y, score, t, perfect_above)
    if metric == "uplift_at_10pct":
        return uplift_at_k(y, score, t, 0.10)
    if metric == "uplift_at_20pct":
        return uplift_at_k(y, score, t, 0.20)
    raise ValueError(f"Unsupported bootstrap metric {metric!r}")


def paired_difference(
    draws: Mapping[str, np.ndarray],
    reference: str,
    comparison: str,
    level: float = 0.95,
) -> dict[str, Any]:
    """Interval for ``reference - comparison`` from paired bootstrap draws."""
    if reference not in draws or comparison not in draws:
        raise KeyError(f"draws must contain both {reference!r} and {comparison!r}")
    difference = np.asarray(draws[reference], dtype=float) - np.asarray(
        draws[comparison], dtype=float
    )
    interval = percentile_interval(difference, level=level)
    finite = difference[np.isfinite(difference)]
    return {
        "reference": reference,
        "comparison": comparison,
        "mean_difference": float(np.mean(finite)) if len(finite) else float("nan"),
        **interval,
        # "Significant" here means the paired interval excludes zero, which is
        # the operational definition of a breakdown used by the regime map.
        "excludes_zero": bool(
            np.isfinite(interval["ci_low"])
            and np.isfinite(interval["ci_high"])
            and (interval["ci_low"] > 0.0 or interval["ci_high"] < 0.0)
        ),
    }


def pool_draws(per_seed: Sequence[Mapping[str, np.ndarray]], model: str) -> np.ndarray:
    """Concatenate one model's draws across seeds into a seed+row distribution."""
    pieces = [np.asarray(draws[model], dtype=float) for draws in per_seed if model in draws]
    if not pieces:
        return np.zeros(0, dtype=float)
    return np.concatenate(pieces)


def bootstrap_predictions_frame(
    predictions: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 0,
    metrics: Sequence[str] = DEFAULT_METRICS,
    max_rows: Optional[int] = None,
    level: float = 0.95,
    cell_keys: Sequence[str] = CELL_KEYS,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, dict[tuple, dict[str, Any]]]:
    """Bootstrap every cell in a concatenated predictions frame.

    Returns a tidy per-(cell, model, metric) table and the raw per-cell results,
    which the analysis script needs for the paired differences.
    """
    keys = [key for key in cell_keys if key in predictions.columns]
    if not keys:
        raise ValueError(f"None of the cell keys {list(cell_keys)} are present")

    rows: list[dict[str, Any]] = []
    per_cell: dict[tuple, dict[str, Any]] = {}
    for cell_key, cell in predictions.groupby(keys, sort=True, dropna=False):
        cell_key = cell_key if isinstance(cell_key, tuple) else (cell_key,)
        if progress is not None:
            progress(f"bootstrap {dict(zip(keys, cell_key))}")
        # Derive the per-cell bootstrap seed from the cell identity so a rerun
        # reproduces the same draws regardless of iteration order. crc32 rather
        # than the builtin hash, which is salted per process.
        cell_seed = int(seed + zlib.crc32("|".join(map(str, cell_key)).encode("utf-8")) % 100_000)
        result = bootstrap_cell(
            cell,
            n_bootstrap=n_bootstrap,
            seed=cell_seed,
            metrics=metrics,
            max_rows=max_rows,
            level=level,
        )
        per_cell[cell_key] = result
        for metric in metrics:
            for model in result["models"]:
                rows.append(
                    {
                        **dict(zip(keys, cell_key)),
                        "model": model,
                        "metric": metric,
                        "point_estimate": result["point"][metric][model],
                        **percentile_interval(result["draws"][metric][model], level=level),
                        "n_rows": result["n_rows"],
                        "n_failed_resamples": result["n_failed_resamples"],
                    }
                )
    return pd.DataFrame(rows), per_cell
