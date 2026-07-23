"""Controlled resampling rules for the three robustness axes.

Grounding
---------
Keith et al., *RCT Rejection Sampling for Causal Estimation Evaluation*
(arXiv:2307.15176), Algorithm 1: accept unit ``i`` with probability

    (1 / M) * P*(T = t_i | C_i) / P(T = t_i),      M >= sup P*(T|C) / P(T)

Their Theorem 3.2 gives identification of the downstream effect; their
Proposition 3.1 is why Gentzel et al.'s OSRCT does not: conditioning on the
selection indicator leaves ``P(C) != P(C | S = 1)`` with no non-parametric
functional recovering the original marginal.

The acceptance rule must never look at ``Y`` for an individual unit. Every
public function here takes only ``t`` and covariate-derived quantities, and
``covariate_selection_sample`` is deliberately unable to see the outcome at all.

Two identification-safe special cases are what the shipped axes use:

``control_share``
    ``P*(T = 1 | C) = pi``, constant in ``C``. Acceptance is uniform inside each
    arm, so ``P(C)`` and ``P(Y | T, C)`` are preserved *and* ``T ⟂ C`` survives:
    the resample is still a valid RCT with a shifted arm ratio.

``conversion``
    Acceptance is a function of ``C`` alone -- never ``T``, never ``Y``. Then
    ``T ⟂ (C, Y(0), Y(1))`` still holds in the accepted sample, i.e. a valid RCT
    on a shifted covariate population. ``P(C)`` moves by design; the CATE stays
    identified with no adjustment.

The general C-dependent sampler is implemented too, so a genuinely confounded
study stays available, but it deliberately breaks ``T ⟂ C`` and therefore makes
Qini stop being a causal quantity. Callers must opt into it explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np


@dataclass
class ResampleRecord:
    """Everything needed to audit one resampling call."""

    rule: str
    params: dict[str, Any] = field(default_factory=dict)
    n_before: int = 0
    n_after: int = 0
    acceptance_rate: float = 1.0
    envelope_M: Optional[float] = None
    treatment_rate_before: Optional[float] = None
    treatment_rate_after: Optional[float] = None
    outcome_rate_before: Optional[float] = None
    outcome_rate_after: Optional[float] = None
    conditions_on_outcome: bool = False
    preserves_randomization: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_rng(rng) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return rng


def _check_treatment(t) -> np.ndarray:
    t = np.asarray(t)
    if t.ndim != 1:
        raise ValueError("treatment must be a 1D vector")
    if len(t) == 0:
        raise ValueError("treatment must not be empty")
    values = set(np.unique(t).tolist())
    if not values.issubset({0, 1}):
        raise ValueError("treatment must be binary and encoded as 0/1")
    return t.astype(np.int8, copy=False)


def _rates(t: np.ndarray, y: Optional[np.ndarray]) -> tuple[float, Optional[float]]:
    treatment_rate = float(np.mean(t)) if len(t) else float("nan")
    outcome_rate = None if y is None else float(np.mean(np.asarray(y, dtype=float)))
    return treatment_rate, outcome_rate


def _finish(
    record: ResampleRecord,
    keep: np.ndarray,
    t: np.ndarray,
    y: Optional[np.ndarray],
) -> tuple[np.ndarray, ResampleRecord]:
    keep = np.sort(np.asarray(keep, dtype=np.int64))
    record.n_before = int(len(t))
    record.n_after = int(len(keep))
    record.acceptance_rate = float(len(keep) / len(t)) if len(t) else 0.0
    before = _rates(t, y)
    after = _rates(t[keep], None if y is None else np.asarray(y)[keep])
    record.treatment_rate_before, record.outcome_rate_before = before
    record.treatment_rate_after, record.outcome_rate_after = after
    return keep, record


# --------------------------------------------------------------------------
# Keith et al. Algorithm 1
# --------------------------------------------------------------------------
def rejection_sample(
    rng: np.random.Generator,
    t,
    propensity_star,
    marginal_propensity: Optional[float] = None,
    envelope_M: Optional[float] = None,
    y=None,
    rule: str = "rct_rejection_sampling",
    params: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, ResampleRecord]:
    """Keith et al. Algorithm 1.

    Parameters
    ----------
    propensity_star
        Target ``P*(T = 1 | C_i)``: a scalar for the constant (non-confounding)
        case, or a per-unit array for the general confounded case.
    marginal_propensity
        The source RCT's ``P(T = 1)``. Estimated from ``t`` when omitted.
    envelope_M
        ``M >= sup P*(T|C) / P(T)``. Computed from the sample when omitted,
        which maximises the accepted sample size.

    ``y`` is used only to *record* the before/after outcome rate. It never
    enters the acceptance probability.
    """
    rng = _check_rng(rng)
    t = _check_treatment(t)

    star = np.asarray(propensity_star, dtype=float)
    if star.ndim == 0:
        star = np.full(len(t), float(star))
    elif star.shape != (len(t),):
        raise ValueError("propensity_star must be a scalar or match the length of t")
    if not np.isfinite(star).all():
        raise ValueError("propensity_star contains NaN or Inf")
    if not ((star > 0.0) & (star < 1.0)).all():
        raise ValueError("propensity_star must satisfy positivity: 0 < P*(T=1|C) < 1")

    if marginal_propensity is None:
        marginal_propensity = float(np.mean(t))
    marginal_propensity = float(marginal_propensity)
    if not 0.0 < marginal_propensity < 1.0:
        raise ValueError("marginal_propensity must lie strictly between 0 and 1")

    # P*(T = t_i | C_i) / P(T = t_i), evaluated at each unit's observed arm.
    star_at_t = np.where(t == 1, star, 1.0 - star)
    marginal_at_t = np.where(t == 1, marginal_propensity, 1.0 - marginal_propensity)
    ratio = star_at_t / marginal_at_t

    supremum = float(ratio.max())
    if envelope_M is None:
        envelope_M = supremum
    envelope_M = float(envelope_M)
    if envelope_M < supremum - 1e-12:
        raise ValueError(
            f"envelope_M={envelope_M} is below the observed sup P*/P={supremum}; "
            "the sampler would not be valid"
        )

    accept_probability = ratio / envelope_M
    keep = np.flatnonzero(rng.random(len(t)) < accept_probability)
    if len(keep) == 0:
        raise ValueError("The rejection sampler accepted no rows; loosen the severity")

    constant_star = bool(np.ptp(star) < 1e-12)
    record = ResampleRecord(
        rule=rule,
        params={
            "propensity_star": float(star[0]) if constant_star else "per-unit",
            "marginal_propensity": marginal_propensity,
            **(params or {}),
        },
        envelope_M=envelope_M,
        conditions_on_outcome=False,
        # A constant P*(T|C) leaves T independent of C; a C-dependent one is
        # exactly the confounding Keith et al. induce on purpose.
        preserves_randomization=constant_star,
    )
    return _finish(record, keep, t, y)


def control_share_rule(
    rng: np.random.Generator,
    t,
    target_treatment_rate: float,
    y=None,
) -> tuple[np.ndarray, ResampleRecord]:
    """Shift the treated/control marginal to ``target_treatment_rate``.

    A constant ``P*(T=1|C)``, routed through :func:`rejection_sample` so there
    is a single acceptance code path. With ``M = sup P*/P`` the arm that has to
    grow in share is kept whole and the other is thinned, which is the largest
    valid sample at the requested ratio.
    """
    rng = _check_rng(rng)
    t = _check_treatment(t)
    target = float(target_treatment_rate)
    if not 0.0 < target < 1.0:
        raise ValueError("target_treatment_rate must lie strictly between 0 and 1")
    if len(np.unique(t)) != 2:
        raise ValueError("control_share_rule needs both treatment arms present")

    return rejection_sample(
        rng,
        t,
        propensity_star=target,
        marginal_propensity=float(np.mean(t)),
        y=y,
        rule="control_share",
        params={"target_treatment_rate": target},
    )


# --------------------------------------------------------------------------
# Covariate-only selection (conversion axis, scale axis)
# --------------------------------------------------------------------------
def covariate_selection_sample(
    rng: np.random.Generator,
    accept_probability,
    t,
    y=None,
    rule: str = "covariate_selection",
    params: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, ResampleRecord]:
    """Accept units with a probability that is a function of covariates only.

    The signature makes the Proposition 3.1 constraint structural: the
    acceptance probability is passed in already computed from ``C``, and ``t``
    and ``y`` are used only for the audit record. Because selection ignores both
    the arm and the outcome, ``T ⟂ (C, Y(0), Y(1))`` survives and the accepted
    sample is a valid RCT on a shifted covariate population.
    """
    rng = _check_rng(rng)
    t = _check_treatment(t)
    probability = np.asarray(accept_probability, dtype=float)
    if probability.shape != (len(t),):
        raise ValueError("accept_probability must have one entry per row")
    if not np.isfinite(probability).all():
        raise ValueError("accept_probability contains NaN or Inf")
    if probability.min() < 0.0 or probability.max() > 1.0:
        raise ValueError("accept_probability must lie in [0, 1]")

    keep = np.flatnonzero(rng.random(len(t)) < probability)
    if len(keep) == 0:
        raise ValueError("Covariate selection accepted no rows; loosen the severity")

    record = ResampleRecord(
        rule=rule,
        params=dict(params or {}),
        conditions_on_outcome=False,
        preserves_randomization=True,
    )
    return _finish(record, keep, t, y)


def conversion_accept_probability(risk_score, lam: float) -> np.ndarray:
    """Acceptance probability that thins high-outcome-risk units.

    ``accept(C) = exp(-lam * r(C))`` where ``r(C)`` is the within-sample
    *rank* of the risk score rescaled to ``[0, 1]``. Using the rank rather than
    the raw score makes the severity knob behave the same way across datasets
    whose risk scores live on very different scales. ``lam = 0`` is the
    identity, and the maximum acceptance probability is always 1, so no sample
    size is thrown away needlessly.
    """
    score = np.asarray(risk_score, dtype=float)
    if score.ndim != 1:
        raise ValueError("risk_score must be a 1D vector")
    if len(score) == 0:
        raise ValueError("risk_score must not be empty")
    if not np.isfinite(score).all():
        raise ValueError("risk_score contains NaN or Inf")
    lam = float(lam)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be a non-negative finite number")
    if lam == 0.0:
        return np.ones(len(score), dtype=float)

    # Average ranks so tied risk scores get identical acceptance probabilities.
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(len(score), dtype=float)
    unique, inverse = np.unique(score, return_inverse=True)
    if len(unique) < len(score):
        sums = np.bincount(inverse, weights=ranks)
        counts = np.bincount(inverse)
        ranks = (sums / counts)[inverse]
    normalized = ranks / max(1.0, float(len(score) - 1))
    return np.exp(-lam * normalized)


def uniform_subsample_rule(
    rng: np.random.Generator,
    t,
    target_rows: Optional[int],
    y=None,
) -> tuple[np.ndarray, ResampleRecord]:
    """Uniformly subsample to ``target_rows`` (the scale axis).

    Uniform selection is a degenerate covariate-only rule: it changes nothing
    about ``P(C)``, ``P(T|C)`` or ``P(Y|T,C)`` in expectation, only the sample
    size. ``target_rows=None`` (or a target at least as large as the input) is
    the undegraded anchor and keeps every row.
    """
    rng = _check_rng(rng)
    t = _check_treatment(t)
    if target_rows is None or int(target_rows) <= 0 or int(target_rows) >= len(t):
        keep = np.arange(len(t), dtype=np.int64)
        record = ResampleRecord(
            rule="scale",
            params={"target_rows": None if target_rows is None else int(target_rows)},
        )
        return _finish(record, keep, t, y)

    target = int(target_rows)
    keep = rng.choice(len(t), size=target, replace=False)
    record = ResampleRecord(rule="scale", params={"target_rows": target})
    return _finish(record, keep, t, y)
