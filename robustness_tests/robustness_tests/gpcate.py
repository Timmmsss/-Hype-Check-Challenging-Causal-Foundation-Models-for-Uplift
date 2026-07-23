"""GP-CATE: the small-control-arm specialist.

arXiv:2605.27473. Each arm's outcome surface gets its own Gaussian process, so
the scarce arm's uncertainty enters the CATE posterior *directly* rather than as
an unmodelled bias. That is the paper's diagnosis of the X-Learner in the
few-placebo regime: its regression target inherits the bias of a nuisance model
fitted to the small arm, which centres the posterior away from the truth and
makes intervals under-cover.

With a constant-times-RBF-plus-white-noise kernel per arm and empirical-Bayes
hyperparameters::

    tau(x) | D  ~  N( m1(x) - m0(x),  s1(x)^2 + s0(x)^2 )

Deviations from a literal reading of the paper, both forced by this repo's data
and documented in ``README_ROBUSTNESS.md``:

1. The paper assumes real-valued ``Y`` and uses a Gaussian likelihood. Our
   outcomes are binary, so this is a linear-probability-style approximation.
   Ranking, which is what Qini scores, is unaffected by the induced
   heteroscedasticity in a way that would change conclusions; calibration of the
   intervals is more affected, and that is stated wherever intervals are used.
2. An isotropic length scale over 60+ one-hot columns (LZD) is meaningless, so
   the features are projected to a handful of principal components first.

Exact GP inference is cubic, so both arms are capped. Following the paper's own
recommendation, the treated arm is capped much harder than the control arm: in
the few-placebo regime the treated arm contributes only a small share of the
CATE variance, and the paper reports that capping it preserves coverage.
"""
from __future__ import annotations

import numpy as np

from . import _paths  # noqa: F401  (side effect: puts baseline_benchmark on sys.path)

from baseline_benchmark.models import _feature_matrix, _training_arrays  # noqa: E402

_PREDICT_BATCH = 4096


class GPCATE:
    """Per-arm Gaussian process CATE with calibrated intervals."""

    name = "gp_cate"

    def __init__(
        self,
        seed: int = 42,
        max_control: int = 2000,
        max_treated: int = 500,
        n_components: int = 20,
        n_restarts: int = 2,
        alpha: float = 1e-8,
    ):
        for label, value in (
            ("max_control", max_control),
            ("max_treated", max_treated),
            ("n_components", n_components),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if isinstance(n_restarts, bool) or not isinstance(n_restarts, (int, np.integer)):
            raise ValueError("n_restarts must be an integer")
        if int(n_restarts) < 0:
            raise ValueError("n_restarts must be non-negative")
        self.seed = int(seed)
        self.max_control = int(max_control)
        self.max_treated = int(max_treated)
        self.n_components = int(n_components)
        self.n_restarts = int(n_restarts)
        self.alpha = float(alpha)

    # ------------------------------------------------------------------
    def _subsample(self, rng: np.random.Generator, n: int, cap: int) -> np.ndarray:
        if n <= cap:
            return np.arange(n)
        return np.sort(rng.choice(n, size=cap, replace=False))

    def _fit_arm(self, X: np.ndarray, y: np.ndarray, seed: int):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        spread = float(np.mean(np.std(X, axis=0))) if X.shape[1] else 1.0
        length_scale = max(spread * np.sqrt(X.shape[1]), 1e-3)
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e4))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts,
            random_state=seed,
        )
        return model.fit(X, y.astype(float))

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        from sklearn.decomposition import PCA

        X, t, y = _training_arrays(X, t, y)
        rng = np.random.default_rng(self.seed)

        n_components = min(self.n_components, X.shape[1], X.shape[0])
        if n_components < X.shape[1]:
            self.projection_ = PCA(n_components=n_components, random_state=self.seed).fit(X)
            X_projected = np.asarray(self.projection_.transform(X), dtype=float)
        else:
            self.projection_ = None
            X_projected = np.asarray(X, dtype=float)

        control = np.flatnonzero(t == 0)
        treated = np.flatnonzero(t == 1)
        control = control[self._subsample(rng, len(control), self.max_control)]
        treated = treated[self._subsample(rng, len(treated), self.max_treated)]
        if len(control) < 2 or len(treated) < 2:
            raise ValueError("gp_cate needs at least two rows in each treatment arm")

        self.gp0_ = self._fit_arm(X_projected[control], y[control], self.seed)
        self.gp1_ = self._fit_arm(X_projected[treated], y[treated], self.seed + 1)
        self.n_features_in_ = X.shape[1]
        self.n_control_used_ = int(len(control))
        self.n_treated_used_ = int(len(treated))
        return self

    # ------------------------------------------------------------------
    def _project(self, X) -> np.ndarray:
        if not hasattr(self, "gp0_"):
            raise RuntimeError("gp_cate must be fitted before predict_cate")
        X = _feature_matrix(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but gp_cate was fitted with "
                f"{self.n_features_in_} features"
            )
        if self.projection_ is None:
            return np.asarray(X, dtype=float)
        return np.asarray(self.projection_.transform(X), dtype=float)

    def _posterior(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation of ``tau(x)``."""
        X_projected = self._project(X)
        means, variances = [], []
        for start in range(0, len(X_projected), _PREDICT_BATCH):
            batch = X_projected[start: start + _PREDICT_BATCH]
            m0, s0 = self.gp0_.predict(batch, return_std=True)
            m1, s1 = self.gp1_.predict(batch, return_std=True)
            means.append(np.asarray(m1, dtype=float) - np.asarray(m0, dtype=float))
            variances.append(np.asarray(s1, dtype=float) ** 2 + np.asarray(s0, dtype=float) ** 2)
        mean = np.concatenate(means)
        sd = np.sqrt(np.concatenate(variances))
        return mean, sd

    def predict_cate(self, X) -> np.ndarray:
        mean, _ = self._posterior(X)
        if not np.isfinite(mean).all():
            raise RuntimeError("gp_cate produced non-finite CATE predictions")
        return mean

    def predict_cate_interval(self, X, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Credible interval for ``tau(x)``; used for the coverage analysis."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie strictly between 0 and 1")
        from statistics import NormalDist

        mean, sd = self._posterior(X)
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        return mean - z * sd, mean + z * sd
