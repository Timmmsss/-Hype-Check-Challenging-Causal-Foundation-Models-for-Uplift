"""Definitions of the three robustness axes and their severity grids.

The axes correspond one-to-one with hypothesis H3 in ``main.tex``: data beyond
the context window (``scale``), a small control group (``control_share``), and
low conversion (``conversion``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

DEGRADE_CHOICES = ("train", "train+val", "all")

#: Models the existing harness already registers in ``baseline_benchmark``.
CORE_MODELS = ("t_learner", "x_learner", "dr_learner", "dragonnet", "causalpfn")


@dataclass(frozen=True)
class AxisSpec:
    name: str
    #: What one severity value means.
    severity_kind: str
    #: Severity values, ordered from undegraded to most severe.
    default_grid: tuple[Any, ...]
    #: Which splits the resampling touches by default.
    default_degrade_splits: str
    #: Specialist estimators that this axis exists to exercise.
    extra_models: tuple[str, ...]
    description: str

    def tag(self, severity: Any) -> str:
        """Directory-safe label for one severity value."""
        if severity is None:
            return f"{self.name}_full"
        if self.severity_kind == "target_rows":
            return f"{self.name}_rows{int(severity)}"
        if self.severity_kind == "treatment_rate":
            return f"{self.name}_trate{float(severity):.3f}".replace(".", "p")
        if self.severity_kind == "conversion_lambda":
            return f"{self.name}_lam{float(severity):.2f}".replace(".", "p")
        raise ValueError(f"Unknown severity kind {self.severity_kind!r}")

    def parse(self, text: str) -> Any:
        """Parse one severity from the command line."""
        value = text.strip().lower()
        if value in {"", "none", "full", "0"} and self.severity_kind == "target_rows":
            return None
        if value in {"", "none", "full"}:
            return None
        if self.severity_kind == "target_rows":
            parsed = int(value)
            if parsed < 0:
                raise ValueError("scale severities must be non-negative row counts")
            return None if parsed == 0 else parsed
        parsed_float = float(value)
        if self.severity_kind == "treatment_rate":
            if not 0.0 < parsed_float < 1.0:
                raise ValueError("control_share severities must lie strictly in (0, 1)")
        if self.severity_kind == "conversion_lambda":
            if parsed_float < 0.0:
                raise ValueError("conversion severities must be non-negative")
        return parsed_float

    def severity_order(self, severity: Any) -> float:
        """Sort key where a larger value means a more degraded regime.

        Used to order the columns of the regime map and to find the first
        severity at which a model's advantage breaks.
        """
        if severity is None:
            return float("-inf")
        if self.severity_kind == "target_rows":
            # Fewer training rows is more severe.
            return -float(severity)
        return float(severity)

    def is_anchor(self, severity: Any) -> bool:
        """True when this severity leaves the data undegraded."""
        if severity is None:
            return True
        if self.severity_kind == "conversion_lambda":
            return float(severity) == 0.0
        return False


AXES: dict[str, AxisSpec] = {
    "scale": AxisSpec(
        name="scale",
        severity_kind="target_rows",
        # None is the anchor: every row of the training split. CausalPFN's
        # default context window is 4096 rows, so the grid straddles it.
        default_grid=(None, 200_000, 50_000, 20_000, 5_000, 2_000),
        default_degrade_splits="train+val",
        extra_models=(),
        description="Training rows available, against a fixed evaluation split.",
    ),
    "control_share": AxisSpec(
        name="control_share",
        severity_kind="treatment_rate",
        # None is the anchor: the dataset's own treatment rate, which differs by
        # dataset (Criteo is heavily treated, RetailHero is near balanced), so
        # the undegraded point cannot be a fixed number. From there the treated
        # share rises, i.e. the control share falls to 1%: the few-placebo end.
        default_grid=(None, 0.7, 0.85, 0.95, 0.98, 0.99),
        default_degrade_splits="train+val",
        extra_models=("gp_cate",),
        description="Treated/control marginal, down to a few-placebo control arm.",
    ),
    "conversion": AxisSpec(
        name="conversion",
        severity_kind="conversion_lambda",
        # Reported by realized base rate; the knob itself is only a means. How
        # far the rate can actually fall is bounded by how well the covariates
        # predict the outcome out of sample, which the precondition diagnostic
        # measures, so the grid reaches well past the point of diminishing
        # returns rather than assuming a fixed reduction.
        default_grid=(0.0, 2.0, 4.0, 6.0, 9.0, 12.0),
        default_degrade_splits="all",
        extra_models=("q_learner", "q_learner_ratio"),
        description="Base conversion rate, via covariate-only selection.",
    ),
}


def get_axis(name: str) -> AxisSpec:
    key = name.strip().lower()
    if key not in AXES:
        raise ValueError(f"Unknown axis {name!r}; choose from {sorted(AXES)}")
    return AXES[key]


def parse_grid(axis: AxisSpec, text: Optional[str]) -> list[Any]:
    """Parse a comma-separated severity grid, or fall back to the default."""
    if text is None or not text.strip():
        return list(axis.default_grid)
    values: list[Any] = []
    for item in text.split(","):
        if not item.strip():
            continue
        parsed = axis.parse(item)
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise ValueError(f"No severities parsed for axis {axis.name!r}")
    return values


def default_models(axis: AxisSpec) -> list[str]:
    """Core baselines plus the specialist this axis was built to test."""
    return [*CORE_MODELS, *axis.extra_models]


def resolve_degrade_splits(axis: AxisSpec, override: Optional[str]) -> tuple[str, ...]:
    """Map the ``--degrade-splits`` choice onto concrete split names."""
    choice = (override or axis.default_degrade_splits).strip().lower()
    if choice not in DEGRADE_CHOICES:
        raise ValueError(f"degrade_splits must be one of {DEGRADE_CHOICES}")
    if choice == "train":
        return ("train",)
    if choice == "train+val":
        return ("train", "validation")
    return ("train", "validation", "test")
