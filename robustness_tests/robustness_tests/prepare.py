"""Build one degraded regime out of a cleaned RCT dataset.

The order of operations is deliberate::

    load_cleaned_dataset()   # reused verbatim, including the global --max-rows cap
    _split_indices()         # reused verbatim: stratified / group-safe, unchanged
    resample indices         # NEW, at raw-index level, on the targeted splits only
    _validate_splits()       # reused
    _make_preprocessor()     # fitted on the RESAMPLED training rows only

Resampling *after* the split means the split protocol and its group-safety
guarantees are inherited untouched, and the preprocessor is still fitted on
training data alone, so no leakage is introduced.
"""
from __future__ import annotations

import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401  (side effect: puts baseline_benchmark on sys.path)
from .axes import AxisSpec, get_axis, resolve_degrade_splits
from .diagnostics import (
    check_event_counts,
    check_outcome_dependent_covariate,
    check_overlap,
    realized_severity,
)
from .resampling import (
    ResampleRecord,
    control_share_rule,
    conversion_accept_probability,
    covariate_selection_sample,
    rejection_sample,
    uniform_subsample_rule,
)

from baseline_benchmark.data import (  # noqa: E402
    PreparedData,
    _make_preprocessor,
    _split_indices,
    _validate_splits,
    load_cleaned_dataset,
)

SPLIT_NAMES = ("train", "validation", "test")


class DegenerateRegimeError(RuntimeError):
    """The requested severity leaves a split unusable for uplift evaluation.

    Raised rather than silently returning a cell that would produce meaningless
    Qini. The runner records it as a failed cell and carries on, because "this
    severity is not estimable" is itself a result.
    """


@dataclass
class ResampledData:
    prepared: PreparedData
    axis: str
    severity: Any
    severity_tag: str
    degrade_splits: tuple[str, ...]
    resample_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    #: Per-split, per-name arrays carried through the resampling (e.g. the true
    #: CATE on semi-synthetic data). Empty for real RCT datasets.
    extras: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    @property
    def realized(self) -> dict[str, float]:
        """Realized severity on the training split."""
        return self.diagnostics.get("realized", {}).get("train", {})

    def true_cate(self, split: str) -> Optional[np.ndarray]:
        """Ground-truth CATE for one split, when the data has it."""
        return self.extras.get(split, {}).get("tau_true")


# --------------------------------------------------------------------------
# Risk score for the conversion axis
# --------------------------------------------------------------------------
def _stable_seed(text: str) -> int:
    """Reproducible across processes, unlike the salted builtin ``hash``."""
    return int(zlib.crc32(text.encode("utf-8")))


def _equalize_target(
    train_rows: int, split_name: str, val_fraction: float, test_fraction: float
) -> int:
    """Scale a training-row budget to the size the other splits should have.

    ``equalize_rows`` is expressed in training rows because that is what a
    reader thinks in. The validation and test splits are trimmed in the same
    proportion as the split protocol, so the ratios stay 60/20/20.
    """
    train_fraction = 1.0 - val_fraction - test_fraction
    if train_fraction <= 0:
        raise ValueError("val_fraction and test_fraction must sum to less than 1")
    if split_name == "train":
        return int(train_rows)
    fraction = val_fraction if split_name == "validation" else test_fraction
    return max(1, int(round(train_rows * fraction / train_fraction)))


def _is_categorical(values: pd.Series) -> bool:
    return values.dtype.kind in "OSUb" or str(values.dtype) in {"category", "string"}


def _ordinal_by_frequency(column: pd.Series) -> np.ndarray:
    """Outcome-free ordinal encoding: rank categories by how common they are."""
    filled = column.astype(object).where(pd.notna(column), "__missing__")
    counts = filled.value_counts()
    ordering = {value: rank for rank, value in enumerate(counts.index)}
    return filled.map(ordering).astype(float).to_numpy()


def _make_fitted_risk_scorer(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    seed: int,
    n_folds: int = 5,
) -> Callable[[np.ndarray], np.ndarray]:
    """``P(Y = 1 | C)`` from a model fitted on the training rows only.

    Selection then depends on covariates through a *fixed function* of ``C``,
    which is what Keith et al.'s Proposition 3.1 permits. The individual outcome
    of a unit being selected is never consulted.

    The training rows are scored **out of fold**. Without that, the training
    split is scored in-sample and every other split out-of-sample, so the same
    severity knob thins training conversions far harder than evaluation
    conversions -- an artifact of model sharpness rather than of the regime. The
    fix costs one extra round of fitting and makes the realized base rate move
    by comparable amounts on every split.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    y_train = np.asarray(y)[train_idx]
    if len(np.unique(y_train)) < 2:
        raise DegenerateRegimeError(
            "The training split has a single outcome class; a conversion risk "
            "score cannot be fitted."
        )

    def _model() -> Any:
        return HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, early_stopping=False, random_state=seed
        )

    preprocessor = _make_preprocessor(X.iloc[train_idx])
    X_train = np.asarray(preprocessor.fit_transform(X.iloc[train_idx]), dtype=np.float32)

    scores = np.full(len(X), np.nan, dtype=float)
    minority = int(min(np.sum(y_train == 0), np.sum(y_train == 1)))
    folds = max(2, min(int(n_folds), minority))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores[train_idx] = cross_val_predict(
        _model(), X_train, y_train, cv=splitter, method="predict_proba"
    )[:, 1]

    full = _model().fit(X_train, y_train)
    positive = int(np.flatnonzero(np.asarray(full.classes_) == 1)[0])
    held_out = np.setdiff1d(np.arange(len(X)), train_idx, assume_unique=False)
    if len(held_out):
        X_held = np.asarray(preprocessor.transform(X.iloc[held_out]), dtype=np.float32)
        scores[held_out] = full.predict_proba(X_held)[:, positive]

    if not np.isfinite(scores).all():
        raise RuntimeError("The conversion risk score left some rows unscored")

    def score(target_idx: np.ndarray) -> np.ndarray:
        return scores[target_idx]

    return score


def _make_natural_risk_scorer(X: pd.DataFrame, column: str) -> Callable[[np.ndarray], np.ndarray]:
    """A single raw covariate, used as the fully outcome-free cross-check."""
    if column not in X.columns:
        raise ValueError(
            f"Conversion risk column {column!r} is not a covariate; "
            f"available={list(X.columns)[:20]}"
        )

    def score(target_idx: np.ndarray) -> np.ndarray:
        values = X.iloc[target_idx][column]
        if _is_categorical(values):
            return _ordinal_by_frequency(values)
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.fillna(numeric.median()).astype(float).to_numpy()

    return score


# --------------------------------------------------------------------------
# Per-split resampling
# --------------------------------------------------------------------------
def _resample_split(
    axis: AxisSpec,
    severity: Any,
    rng: np.random.Generator,
    split_idx: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    risk_score: Optional[np.ndarray],
    confounding_propensity: Optional[np.ndarray],
) -> tuple[np.ndarray, ResampleRecord]:
    """Apply this axis's rule to one split, returning global row indices."""
    t_split, y_split = t[split_idx], y[split_idx]

    if confounding_propensity is not None:
        local, record = rejection_sample(
            rng,
            t_split,
            propensity_star=confounding_propensity,
            y=y_split,
            rule="confounded_rejection_sampling",
        )
    elif axis.name == "scale":
        local, record = uniform_subsample_rule(rng, t_split, severity, y=y_split)
    elif axis.name == "control_share":
        local, record = control_share_rule(rng, t_split, float(severity), y=y_split)
    elif axis.name == "conversion":
        if risk_score is None:
            raise ValueError("The conversion axis needs a risk score")
        probability = conversion_accept_probability(risk_score, float(severity))
        local, record = covariate_selection_sample(
            rng,
            probability,
            t_split,
            y=y_split,
            rule="conversion",
            params={"lam": float(severity)},
        )
    else:
        raise ValueError(f"No resampling rule for axis {axis.name!r}")

    return split_idx[local], record


def prepare_resampled_data(
    cleaned_root: Path,
    dataset: str,
    axis: str = "scale",
    severity: Any = None,
    outcome: Optional[str] = None,
    max_rows: Optional[int] = None,
    seed: int = 42,
    **kwargs: Any,
) -> ResampledData:
    """Prepare one (dataset, axis, severity, seed) cell from cleaned Parquet data.

    ``max_rows`` is the *global* pre-split cap inherited from the existing
    harness (needed to make Criteo tractable at all). It is independent of the
    scale axis, whose severity subsamples the training split after the split is
    drawn, so that the evaluation split is byte-identical across severities.
    """
    X, t, y, ids, outcome_name, group_safe = load_cleaned_dataset(
        cleaned_root=cleaned_root,
        dataset=dataset,
        outcome=outcome,
        max_rows=max_rows,
        seed=seed,
    )
    return build_resampled_split(
        X=X,
        t=t,
        y=y,
        ids=ids,
        dataset=dataset,
        outcome_name=outcome_name,
        group_safe=group_safe,
        axis=axis,
        severity=severity,
        seed=seed,
        **kwargs,
    )


def build_resampled_split(
    X: pd.DataFrame,
    t: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    dataset: str,
    outcome_name: str,
    group_safe: bool,
    axis: str = "scale",
    severity: Any = None,
    seed: int = 42,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    degrade_splits: Optional[str] = None,
    conversion_risk: str = "fitted",
    conversion_risk_sign: float = 1.0,
    run_diagnostics: bool = True,
    strict_precondition: bool = True,
    confounding_fn: Optional[Any] = None,
    allow_confounding: bool = False,
    equalize_rows: Optional[int] = None,
    extra_arrays: Optional[dict[str, np.ndarray]] = None,
) -> ResampledData:
    """Split, resample and preprocess in-memory data.

    Shared by the real-RCT loader above and by the semi-synthetic generator, so
    both go through exactly the same split protocol, resampling rules and
    train-only preprocessing. ``extra_arrays`` carries per-row quantities that
    are not features -- the true CATE, above all -- through the resampling and
    splitting, so they stay aligned with the returned splits.
    """
    axis_spec = get_axis(axis)
    targeted = resolve_degrade_splits(axis_spec, degrade_splits)
    if confounding_fn is not None and not allow_confounding:
        raise ValueError(
            "A covariate-dependent P*(T|C) induces confounding, which makes Qini "
            "stop being a causal quantity. Pass allow_confounding=True to opt in."
        )
    extra_arrays = dict(extra_arrays or {})
    for name, values in extra_arrays.items():
        if len(values) != len(t):
            raise ValueError(f"extra array {name!r} must have one entry per row")

    train_idx, val_idx, test_idx = _split_indices(
        X=X,
        t=t,
        y=y,
        group_safe=group_safe,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    split_indices = {"train": train_idx, "validation": val_idx, "test": test_idx}

    diagnostics: dict[str, Any] = {
        "axis": axis_spec.name,
        "severity": severity,
        "degrade_splits": list(targeted),
        "anchor": axis_spec.is_anchor(severity),
    }

    # Keith et al. §4.4 precondition, checked on the untouched training split.
    if run_diagnostics and axis_spec.name == "conversion":
        precondition_preprocessor = _make_preprocessor(X.iloc[train_idx])
        X_train_numeric = np.asarray(
            precondition_preprocessor.fit_transform(X.iloc[train_idx]), dtype=np.float32
        )
        diagnostics["outcome_dependent_covariate"] = check_outcome_dependent_covariate(
            X_train_numeric,
            y[train_idx],
            feature_names=precondition_preprocessor.get_feature_names_out().tolist(),
            strict=strict_precondition,
        )

    # The conversion risk model is fitted once, on the training rows, and then
    # applied to every targeted split.
    risk_scorer: Optional[Callable[[np.ndarray], np.ndarray]] = None
    if axis_spec.name == "conversion" and not axis_spec.is_anchor(severity):
        if conversion_risk == "fitted":
            risk_scorer = _make_fitted_risk_scorer(X, y, train_idx, seed)
        elif conversion_risk.startswith("natural:"):
            risk_scorer = _make_natural_risk_scorer(X, conversion_risk.split(":", 1)[1])
        else:
            raise ValueError(
                "conversion_risk must be 'fitted' or 'natural:<column>'; "
                f"got {conversion_risk!r}"
            )

    # Resample each targeted split with its own generator stream, so a change of
    # severity on one split cannot shift the draw on another.
    resampled: dict[str, np.ndarray] = dict(split_indices)
    records: dict[str, dict[str, Any]] = {}
    for position, split_name in enumerate(SPLIT_NAMES):
        idx = split_indices[split_name]
        if split_name not in targeted or axis_spec.is_anchor(severity):
            records[split_name] = ResampleRecord(
                rule=f"{axis_spec.name}:untouched",
                params={"severity": severity},
                n_before=int(len(idx)),
                n_after=int(len(idx)),
                treatment_rate_before=float(np.mean(t[idx])),
                treatment_rate_after=float(np.mean(t[idx])),
                outcome_rate_before=float(np.mean(y[idx])),
                outcome_rate_after=float(np.mean(y[idx])),
            ).as_dict()
            continue

        rng = np.random.default_rng([seed, position, _stable_seed(axis_spec.tag(severity))])

        risk_score = None
        if risk_scorer is not None:
            risk_score = float(conversion_risk_sign) * risk_scorer(idx)

        confounding_propensity = None
        if confounding_fn is not None:
            confounding_propensity = np.asarray(confounding_fn(X.iloc[idx]), dtype=float)

        kept, record = _resample_split(
            axis_spec, severity, rng, idx, t, y, risk_score, confounding_propensity
        )
        record.params.setdefault("conversion_risk", conversion_risk)

        if equalize_rows:
            # A selection rule that thins hard also shrinks the sample, which
            # would confound this axis with the scale axis. Trimming every
            # severity to a common size leaves the regime as the only thing
            # that varies. The trim is uniform, so it changes nothing else about
            # the distribution.
            target = _equalize_target(
                equalize_rows, split_name, val_fraction, test_fraction
            )
            local, trim = uniform_subsample_rule(rng, t[kept], target, y=y[kept])
            kept = kept[local]
            record.params["equalize_rows_train"] = int(equalize_rows)
            record.params["equalize_rows_target"] = int(target)
            record.params["n_after_equalization"] = int(trim.n_after)
            # Equalization is a cap, not a guarantee: a harsh severity can yield
            # fewer rows than the target, in which case sample size still varies
            # along the axis. Surfaced so the analysis can flag it rather than
            # quietly reporting a confounded comparison.
            record.params["equalize_reached"] = bool(len(kept) >= target)
            record.n_after = int(len(kept))
            record.acceptance_rate = float(len(kept) / len(idx)) if len(idx) else 0.0
            record.treatment_rate_after = float(np.mean(t[kept]))
            record.outcome_rate_after = float(np.mean(y[kept]))

        resampled[split_name] = kept
        records[split_name] = record.as_dict()

    keep = np.concatenate([resampled[name] for name in SPLIT_NAMES])
    order = np.argsort(keep, kind="mergesort")
    keep = keep[order]
    position_of = {int(value): i for i, value in enumerate(keep)}
    local = {
        name: np.array([position_of[int(v)] for v in resampled[name]], dtype=np.int64)
        for name in SPLIT_NAMES
    }

    t_keep, y_keep, ids_keep = t[keep], y[keep], ids[keep]
    X_keep = X.iloc[keep].reset_index(drop=True)

    # Event-count gating before the reused validator, so the failure message
    # names the regime rather than a generic split problem.
    event_checks = {
        name: check_event_counts(t_keep[local[name]], y_keep[local[name]], name)
        for name in SPLIT_NAMES
    }
    diagnostics["event_counts"] = event_checks
    diagnostics["realized"] = {
        name: realized_severity(t_keep[local[name]], y_keep[local[name]])
        for name in SPLIT_NAMES
    }
    unusable = [name for name, check in event_checks.items() if not check["both_arms_present"]]
    if unusable:
        raise DegenerateRegimeError(
            f"axis={axis_spec.name} severity={severity}: split(s) {unusable} lost a "
            "treatment arm entirely; this severity is not estimable."
        )
    diagnostics["degenerate"] = any(check["degenerate"] for check in event_checks.values())
    diagnostics["equalization_reached"] = all(
        record.get("params", {}).get("equalize_reached", True) for record in records.values()
    )

    with warnings.catch_warnings():
        # data.py warns below ten events per arm; check_event_counts already
        # captured that verdict in a structured form.
        warnings.simplefilter("ignore", RuntimeWarning)
        _validate_splits(local["train"], local["validation"], local["test"], t_keep, y_keep)

    preprocessor = _make_preprocessor(X_keep.iloc[local["train"]])
    X_train = np.asarray(
        preprocessor.fit_transform(X_keep.iloc[local["train"]]), dtype=np.float32
    )
    X_val = np.asarray(preprocessor.transform(X_keep.iloc[local["validation"]]), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(X_keep.iloc[local["test"]]), dtype=np.float32)
    for name, matrix in (("train", X_train), ("val", X_val), ("test", X_test)):
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite values remain in transformed {name} features")

    if run_diagnostics and axis_spec.name == "control_share":
        diagnostics["overlap"] = check_overlap(X_train, t_keep[local["train"]], seed=seed)

    split_column = np.full(len(keep), "", dtype=object)
    for name in SPLIT_NAMES:
        split_column[local[name]] = name
    split_table = pd.DataFrame({"epk_id": ids_keep, "split": split_column})

    prepared = PreparedData(
        dataset=dataset.lower(),
        outcome=outcome_name,
        feature_names=preprocessor.get_feature_names_out().tolist(),
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        t_train=t_keep[local["train"]],
        t_val=t_keep[local["validation"]],
        t_test=t_keep[local["test"]],
        y_train=y_keep[local["train"]],
        y_val=y_keep[local["validation"]],
        y_test=y_keep[local["test"]],
        id_train=ids_keep[local["train"]],
        id_val=ids_keep[local["validation"]],
        id_test=ids_keep[local["test"]],
        split_table=split_table,
        preprocessor=preprocessor,
        group_safe=group_safe,
    )
    split_extras = {
        name: {
            key: np.asarray(values)[keep][local[name]]
            for key, values in extra_arrays.items()
        }
        for name in SPLIT_NAMES
    }
    return ResampledData(
        prepared=prepared,
        axis=axis_spec.name,
        severity=severity,
        severity_tag=axis_spec.tag(severity),
        degrade_splits=targeted,
        resample_records=records,
        diagnostics=diagnostics,
        extras=split_extras,
    )
