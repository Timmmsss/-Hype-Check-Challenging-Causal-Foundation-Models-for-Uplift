"""Model registry for the robustness study.

The two 2026 specialists are deliberately *not* added to
``baseline_benchmark.models.TRADITIONAL_MODELS``:
``tests/test_smoke.py::test_available_model_set_includes_causalpfn`` asserts
that ``available_models()`` equals an exact five-element list, so registering
there would break a passing test. Delegation keeps that contract intact while
giving the runner one uniform factory.
"""
from __future__ import annotations

from typing import Any

from . import _paths  # noqa: F401  (side effect: puts baseline_benchmark on sys.path)

from baseline_benchmark.models import available_models, make_model  # noqa: E402

#: Estimators added by this package. Values are import paths, resolved lazily so
#: that importing the registry never pulls in sklearn's GP machinery.
ROBUSTNESS_MODELS: dict[str, tuple[str, str]] = {
    "q_learner": (".qlearner", "QLearner"),
    "q_learner_ratio": (".qlearner", "QLearnerRatio"),
    "gp_cate": (".gpcate", "GPCATE"),
}

#: Constructor arguments accepted per new model, mirroring ``make_model``'s
#: filtering so the runner can pass one wide kwargs bag to everything.
_ALLOWED_KWARGS: dict[str, set[str]] = {
    "q_learner": {"seed", "max_iter", "max_leaf_nodes", "learning_rate", "min_converters"},
    "q_learner_ratio": {"seed", "max_iter", "max_leaf_nodes", "learning_rate", "min_converters"},
    "gp_cate": {
        "seed",
        "max_control",
        "max_treated",
        "n_components",
        "n_restarts",
        "alpha",
    },
}


def _load(name: str):
    from importlib import import_module

    module_name, class_name = ROBUSTNESS_MODELS[name]
    module = import_module(module_name, package=__package__)
    return getattr(module, class_name)


def available_robustness_models() -> list[str]:
    """Every model the robustness runner can build."""
    return [*available_models(), *sorted(ROBUSTNESS_MODELS)]


def make_robustness_model(name: str, **kwargs: Any):
    """Build any baseline or robustness model from one keyword bag."""
    key = name.strip().lower()
    if key in ROBUSTNESS_MODELS:
        allowed = _ALLOWED_KWARGS[key]
        return _load(key)(**{k: v for k, v in kwargs.items() if k in allowed})
    if key in available_models():
        return make_model(key, **kwargs)
    raise ValueError(
        f"Unknown model {name!r}; choose from {available_robustness_models()}"
    )


#: The single hyperparameter each swept model varies along, with the constructor
#: keyword it maps to. The robustness runner reads this so one ``--*-variants``
#: flag per model expands into per-value fits without hard-coding the mapping in
#: three places. ``max_context_length`` is CausalPFN's, forwarded through
#: ``make_model``'s own kwargs filter.
SWEEPABLE_HYPERPARAMETERS: dict[str, str] = {
    "causalpfn": "max_context_length",
    "gp_cate": "n_components",
    "q_learner": "min_converters",
    "q_learner_ratio": "min_converters",
}


def describe_hyperparameters(model: Any, name: str) -> dict[str, Any]:
    """Best-effort record of the hyperparameters a fitted model actually used.

    Prefers the estimator's own ``hyperparameters()`` (the robustness specialists
    expose one, reporting the *realized* settings after any internal clamping),
    and otherwise falls back to reading the public constructor attributes off the
    baseline estimators. Returned as plain JSON-safe scalars so the runner can
    drop it straight into ``metrics.csv``.
    """
    if hasattr(model, "hyperparameters") and callable(model.hyperparameters):
        try:
            raw = dict(model.hyperparameters())
        except Exception:  # noqa: BLE001 - metadata must never break a fit
            raw = {}
    else:
        keys = (
            "seed", "max_iter", "max_leaf_nodes", "learning_rate", "n_folds",
            "epochs", "batch_size", "hidden_dim", "patience",
            "max_context_length", "max_query_length", "num_neighbours",
        )
        raw = {key: getattr(model, key) for key in keys if hasattr(model, key)}
    scalar: dict[str, Any] = {}
    for key, value in raw.items():
        scalar[key] = value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
    scalar.setdefault("model", name)
    return scalar


def supports_pehe(name: str) -> bool:
    """Whether a model's score is on the difference-CATE scale.

    ``q_learner_ratio`` returns ``E[Y|T=1,x] / E[Y|T=0,x]``, a perfectly good
    targeting score but not comparable to a difference-scale ground truth, so it
    is reported as NaN in PEHE tables.
    """
    return name.strip().lower() != "q_learner_ratio"
