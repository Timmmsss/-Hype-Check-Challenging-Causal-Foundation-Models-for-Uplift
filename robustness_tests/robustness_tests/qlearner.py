"""Q-Learner: the low-conversion specialist.

Fuchs & Kreiss, *Beyond Differences: Doubly Robust Meta-Learners for
Ratio-Based Treatment Effects* (arXiv:2605.26288).

A Bayes-rule identity factorises the ratio-CATE into a product of two odds
ratios::

    tau_ratio(x) = mu1(x) / mu0(x) = [p(x) / (1 - p(x))] * [(1 - e(x)) / e(x)]

with ``e(x) = P(W = 1 | x)`` the ordinary propensity and
``p(x) = P(W = 1 | Y = 1, x)`` the *converter* propensity. It follows from
``mu1(x) = p(x) m(x) / e(x)`` and ``mu0(x) = (1 - p(x)) m(x) / (1 - e(x))``,
where the marginal conversion probability ``m(x) = P(Y = 1 | x)`` cancels.

Both nuisances are *classifications*, never an outcome regression on a rare
event, which is exactly why the estimator survives low conversion -- the regime
the ``conversion`` axis manufactures.

Two estimators are exposed because the two output scales are not
interchangeable:

``QLearner``
    Difference scale ``mu1 - mu0``, recovered with a third classifier for
    ``m(x)``. This is the default, so the estimator drops straight into the
    existing Qini metrics *and* into the PEHE table alongside everything else.

``QLearnerRatio``
    The paper's native ratio score. A valid targeting score -- the paper itself
    evaluates it with Qini -- but not comparable to a difference-scale ground
    truth, so PEHE is reported as NaN for it.
"""
from __future__ import annotations

import numpy as np

from . import _paths  # noqa: F401  (side effect: puts baseline_benchmark on sys.path)

from baseline_benchmark.models import (  # noqa: E402
    _classifier,
    _feature_matrix,
    _fit_classifier,
    _positive_probability,
    _training_arrays,
)

#: Clipping bounds from the paper's Appendix A.
PROPENSITY_CLIP = (0.01, 0.99)
RATIO_CLIP = (0.01, 100.0)


class _QLearnerBase:
    """Shared fitting logic for the ratio and difference variants."""

    scale = "difference"

    def __init__(
        self,
        seed: int = 42,
        max_iter: int = 150,
        max_leaf_nodes: int = 31,
        learning_rate: float = 0.05,
        min_converters: int = 50,
    ):
        if isinstance(min_converters, bool) or not isinstance(min_converters, (int, np.integer)):
            raise ValueError("min_converters must be an integer")
        if int(min_converters) < 2:
            raise ValueError("min_converters must be at least 2")
        self.seed = int(seed)
        self.min_converters = int(min_converters)
        # Same base learner family as the classical core in models.py, so the
        # comparison is about the estimator and not about model capacity.
        self.propensity_model = _classifier(seed, max_iter, max_leaf_nodes, learning_rate)
        self.converter_model = _classifier(seed + 1, max_iter, max_leaf_nodes, learning_rate)
        self.marginal_model = _classifier(seed + 2, max_iter, max_leaf_nodes, learning_rate)

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        X, t, y = _training_arrays(X, t, y)

        converters = y == 1
        n_converters = int(converters.sum())
        if n_converters < self.min_converters:
            raise ValueError(
                f"q_learner needs at least {self.min_converters} converters to estimate "
                f"P(W=1|Y=1,X); this sample has {n_converters}."
            )
        converter_arms = set(np.unique(t[converters]).tolist())
        if converter_arms != {0, 1}:
            raise ValueError(
                "q_learner needs converters in both treatment arms; this sample has "
                f"converters only in arm(s) {sorted(converter_arms)}."
            )

        # Stage 1: propensity e(x) = P(W = 1 | x), on every row.
        self.propensity_model = _fit_classifier(self.propensity_model, X, t)
        # Stage 2: converter propensity p(x) = P(W = 1 | Y = 1, x), converters only.
        self.converter_model = _fit_classifier(
            self.converter_model, X[converters], t[converters]
        )
        # Stage 3 (difference scale only): marginal conversion m(x) = P(Y = 1 | x).
        if self.scale == "difference":
            self.marginal_model = _fit_classifier(self.marginal_model, X, y)

        self.n_features_in_ = X.shape[1]
        self.n_converters_ = n_converters
        return self

    def hyperparameters(self) -> dict:
        """The hyperparameters this estimator was built and fitted with.

        Logged into the sweep's ``metrics.csv`` so a robustness run over several
        ``min_converters`` thresholds records, per cell, which threshold produced
        each row and how many converters the degraded sample actually had -- the
        quantity ``min_converters`` gates on. When ``n_converters`` sits just above
        the threshold, the converter propensity ``p(x)`` is fitted on very few
        rows, which is exactly the fragility the conversion axis is probing.
        """
        return {
            "model": self.name,
            "seed": self.seed,
            "scale": self.scale,
            "min_converters": self.min_converters,
            "n_converters": getattr(self, "n_converters_", None),
        }

    def _nuisances(self, X) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError(f"{self.name} must be fitted before predict_cate")
        X = _feature_matrix(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.name} was fitted with "
                f"{self.n_features_in_} features"
            )
        low, high = PROPENSITY_CLIP
        e = np.clip(_positive_probability(self.propensity_model, X), low, high)
        p = np.clip(_positive_probability(self.converter_model, X), low, high)
        return e, p

    def predict_ratio(self, X) -> np.ndarray:
        """The paper's native estimand ``mu1 / mu0``."""
        e, p = self._nuisances(X)
        ratio = (p / (1.0 - p)) * ((1.0 - e) / e)
        return np.clip(ratio, *RATIO_CLIP)


class QLearner(_QLearnerBase):
    """Difference-scale Q-Learner: ``mu1(x) - mu0(x)``."""

    name = "q_learner"
    scale = "difference"

    def predict_cate(self, X) -> np.ndarray:
        e, p = self._nuisances(X)
        X = _feature_matrix(X, name="X")
        m = _positive_probability(self.marginal_model, X)
        mu1 = np.clip(m * p / e, 0.0, 1.0)
        mu0 = np.clip(m * (1.0 - p) / (1.0 - e), 0.0, 1.0)
        cate = mu1 - mu0
        if not np.isfinite(cate).all():
            raise RuntimeError("q_learner produced non-finite CATE predictions")
        return np.asarray(cate, dtype=float)


class QLearnerRatio(_QLearnerBase):
    """Ratio-scale Q-Learner: ``mu1(x) / mu0(x)``, for ranking only."""

    name = "q_learner_ratio"
    scale = "ratio"

    def predict_cate(self, X) -> np.ndarray:
        ratio = self.predict_ratio(X)
        if not np.isfinite(ratio).all():
            raise RuntimeError("q_learner_ratio produced non-finite predictions")
        return np.asarray(ratio, dtype=float)
