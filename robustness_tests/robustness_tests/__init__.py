"""Robustness tests (project goal G2): map where CausalPFN's ranking advantage breaks.

Three axes, matching hypothesis H3 in ``main.tex``: data beyond the context
window (``scale``), a small control group (``control_share``), and low
conversion (``conversion``).

The package sits alongside ``baseline_benchmark`` and reuses its data pipeline,
metrics and estimators rather than reimplementing them.

Top-level names are resolved lazily so that the light modules -- the resampling
rules, the axis definitions, the statistics -- stay importable without sklearn,
torch or the CausalPFN weights being present.
"""
from __future__ import annotations

from typing import Any

from . import _paths  # noqa: F401  (side effect: puts baseline_benchmark on sys.path)
from .axes import AXES, AxisSpec, get_axis
from .resampling import ResampleRecord

_LAZY: dict[str, str] = {
    "DegenerateRegimeError": ".prepare",
    "ResampledData": ".prepare",
    "prepare_resampled_data": ".prepare",
    "available_robustness_models": ".registry",
    "make_robustness_model": ".registry",
    "supports_pehe": ".registry",
}

__all__ = [
    "AXES",
    "AxisSpec",
    "DegenerateRegimeError",
    "ResampleRecord",
    "ResampledData",
    "available_robustness_models",
    "get_axis",
    "make_robustness_model",
    "prepare_resampled_data",
    "supports_pehe",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], package=__name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
