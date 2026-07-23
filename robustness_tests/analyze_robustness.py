#!/usr/bin/env python
"""Turn a robustness sweep into the regime-breakdown map (project goal G2).

Produces, per axis:

1. ``qini_bootstrap_<axis>.csv`` -- per (dataset, severity, model) the mean
   metric with a percentile bootstrap interval, pooled over seeds.
2. ``paired_<axis>.csv`` -- the paired ``CausalPFN - best alternative``
   difference with its interval. The difference, not the levels, is what H3
   claims, and pairing on identical resampled rows is what makes it valid.
3. ``breakpoint_<axis>.csv`` -- the least severe setting at which that paired
   interval first excludes zero on the losing side. This is the handoff
   artifact for G3 (CausalPFN-Rank).
4. ``pehe_<axis>.csv`` and ``spearman_<axis>.csv`` -- semi-synthetic only:
   effect accuracy, and the Spearman agreement between the PEHE ranking and the
   Qini ranking *inside* each degraded regime (H2).
5. ``regime_map.md`` and, if matplotlib is installed, one figure per axis.

It reads the artifacts the sweep already wrote and never refits a model, so it
also runs against the completed part-1/2 ``predictions.parquet`` files.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robustness_tests.axes import AXES, get_axis  # noqa: E402
from robustness_tests.bootstrap import (  # noqa: E402
    DEFAULT_METRICS,
    bootstrap_cell,
    percentile_interval,
)
from robustness_tests.pehe import ranking_agreement  # noqa: E402

REFERENCE_MODEL = "causalpfn"
CELL_KEYS = ("dataset", "axis", "severity_tag", "seed")
#: Separator between a base model name and its hyperparameter-variant tag in the
#: series id, e.g. ``causalpfn#ctx1024``. The sweep writes it; the analysis splits
#: on it to recover which model a variant belongs to.
VARIANT_SEP = "#"


def base_model(model_id: str) -> str:
    """The base estimator a (possibly hyperparameter-suffixed) series belongs to."""
    return str(model_id).split(VARIANT_SEP, 1)[0]


def hparam_tag(model_id: str) -> str:
    """The variant tag of a series id, or ``'default'`` for the plain base name."""
    parts = str(model_id).split(VARIANT_SEP, 1)
    return parts[1] if len(parts) == 2 else "default"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the regime-breakdown map.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results_robustness",
        help="Directory holding the sweep output (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <results-root>/analysis.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--bootstrap-max-rows",
        type=int,
        default=0,
        help="Subsample large evaluation splits before bootstrapping. 0 means all rows.",
    )
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--reference-model", default=REFERENCE_MODEL)
    parser.add_argument(
        "--no-figures", action="store_true", help="Skip matplotlib figures even if available."
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_metrics(results_root: Path) -> pd.DataFrame:
    files = sorted(results_root.glob("**/metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics.csv found under {results_root}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source"] = str(path)
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    # Runs produced before the robustness sweep have no axis columns; treat them
    # as the undegraded anchor so existing results can be compared directly.
    for column, default in (
        ("axis", "baseline"),
        ("severity_tag", "baseline_full"),
        ("severity", "full"),
        ("status", "ok"),
    ):
        if column not in metrics.columns:
            metrics[column] = default
        metrics[column] = metrics[column].fillna(default)
    return metrics


def bootstrap_all_cells(
    results_root: Path,
    metrics: list[str],
    n_bootstrap: int,
    seed: int,
    max_rows: Optional[int],
    level: float,
) -> tuple[pd.DataFrame, dict[tuple, dict[str, Any]]]:
    """Bootstrap each prediction file in turn, keeping memory bounded."""
    files = sorted(results_root.glob("**/predictions.parquet"))
    if not files:
        raise FileNotFoundError(f"No predictions.parquet found under {results_root}")

    rows: list[dict[str, Any]] = []
    per_cell: dict[tuple, dict[str, Any]] = {}
    for path in files:
        frame = pd.read_parquet(path)
        for column, default in (("axis", "baseline"), ("severity_tag", "baseline_full")):
            if column not in frame.columns:
                frame[column] = default
        keys = [key for key in CELL_KEYS if key in frame.columns]
        for cell_key, cell in frame.groupby(keys, sort=True, dropna=False):
            cell_key = cell_key if isinstance(cell_key, tuple) else (cell_key,)
            try:
                result = bootstrap_cell(
                    cell,
                    n_bootstrap=n_bootstrap,
                    # crc32, not the builtin hash, which is salted per process.
                    seed=seed + int(zlib.crc32(str(cell_key).encode("utf-8")) % 1000),
                    metrics=metrics,
                    max_rows=max_rows,
                    level=level,
                )
            except ValueError as exc:
                print(f"SKIP bootstrap for {cell_key}: {exc}")
                continue
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


# --------------------------------------------------------------------------
# Pooling and comparisons
# --------------------------------------------------------------------------
def pool_across_seeds(
    per_cell: dict[tuple, dict[str, Any]],
    metric: str,
    level: float,
) -> pd.DataFrame:
    """Pool per-seed bootstrap draws so the interval carries row *and* split noise."""
    grouped: dict[tuple[str, str, str], dict[str, list[np.ndarray]]] = {}
    for (dataset, axis, severity_tag, _seed), result in per_cell.items():
        key = (dataset, axis, severity_tag)
        bucket = grouped.setdefault(key, {})
        for model in result["models"]:
            bucket.setdefault(model, []).append(result["draws"][metric][model])

    rows = []
    for (dataset, axis, severity_tag), bucket in grouped.items():
        for model, pieces in bucket.items():
            pooled = np.concatenate(pieces)
            finite = pooled[np.isfinite(pooled)]
            rows.append(
                {
                    "dataset": dataset,
                    "axis": axis,
                    "severity_tag": severity_tag,
                    "model": model,
                    "metric": metric,
                    "n_seeds": len(pieces),
                    "mean": float(np.mean(finite)) if len(finite) else float("nan"),
                    **percentile_interval(pooled, level=level),
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(
    per_cell: dict[tuple, dict[str, Any]],
    metric: str,
    reference: str,
    level: float,
) -> pd.DataFrame:
    """``reference - alternative`` intervals, paired within a seed then pooled.

    ``best_alternative`` is chosen on the pooled point estimate, i.e. the
    strongest competitor a practitioner would actually have used, and the
    difference against it is the operational form of H3.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, np.ndarray]]] = {}
    for (dataset, axis, severity_tag, _seed), result in per_cell.items():
        grouped.setdefault((dataset, axis, severity_tag), []).append(result["draws"][metric])

    rows = []
    for (dataset, axis, severity_tag), seed_draws in grouped.items():
        models = sorted({model for draws in seed_draws for model in draws})
        if reference not in models:
            continue
        # A hyperparameter variant is never its own competitor: comparing
        # causalpfn#ctx1024 against causalpfn#ctx4096 would answer a different
        # question than "does this configuration still beat the baselines". The
        # standard (unsuffixed) case has no same-base siblings, so this is a
        # no-op there.
        reference_base = base_model(reference)
        alternatives = [
            model
            for model in models
            if model != reference and base_model(model) != reference_base
        ]
        if not alternatives:
            continue

        pooled_mean: dict[str, float] = {}
        for model in models:
            values = np.concatenate(
                [draws[model] for draws in seed_draws if model in draws]
            )
            finite = values[np.isfinite(values)]
            pooled_mean[model] = float(np.mean(finite)) if len(finite) else float("nan")

        usable = [m for m in alternatives if np.isfinite(pooled_mean[m])]
        best = max(usable, key=lambda m: pooled_mean[m]) if usable else None
        targets = [*alternatives, *(["__best__"] if best else [])]

        for target in targets:
            comparison = best if target == "__best__" else target
            differences = []
            for draws in seed_draws:
                if reference not in draws or comparison not in draws:
                    continue
                differences.append(
                    np.asarray(draws[reference], dtype=float)
                    - np.asarray(draws[comparison], dtype=float)
                )
            if not differences:
                continue
            pooled = np.concatenate(differences)
            finite = pooled[np.isfinite(pooled)]
            interval = percentile_interval(pooled, level=level)
            rows.append(
                {
                    "dataset": dataset,
                    "axis": axis,
                    "severity_tag": severity_tag,
                    "metric": metric,
                    "reference": reference,
                    "comparison": "best_alternative" if target == "__best__" else comparison,
                    "comparison_model": comparison,
                    "mean_difference": float(np.mean(finite)) if len(finite) else float("nan"),
                    **interval,
                    "excludes_zero": bool(
                        np.isfinite(interval["ci_low"])
                        and np.isfinite(interval["ci_high"])
                        and (interval["ci_low"] > 0.0 or interval["ci_high"] < 0.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def find_breakpoints(paired: pd.DataFrame, metrics_frame: pd.DataFrame) -> pd.DataFrame:
    """First severity where the reference model's advantage is significantly negative."""
    if paired.empty:
        return pd.DataFrame()

    severity_by_tag = (
        metrics_frame.dropna(subset=["severity_tag"])
        .drop_duplicates(subset=["axis", "severity_tag"])
        .set_index(["axis", "severity_tag"])["severity"]
        .to_dict()
    )

    rows = []
    best_only = paired.loc[paired["comparison"] == "best_alternative"]
    for (dataset, axis, metric), group in best_only.groupby(["dataset", "axis", "metric"]):
        if axis not in AXES:
            continue
        spec = get_axis(axis)

        def order(tag: str) -> float:
            severity = severity_by_tag.get((axis, tag))
            if severity is None or (isinstance(severity, str) and severity == "full"):
                return spec.severity_order(None)
            try:
                return spec.severity_order(float(severity))
            except (TypeError, ValueError):
                return float("inf")

        ordered = group.assign(_order=group["severity_tag"].map(order)).sort_values("_order")
        losing = ordered.loc[ordered["excludes_zero"] & (ordered["mean_difference"] < 0.0)]
        first = losing.iloc[0] if len(losing) else None
        rows.append(
            {
                "dataset": dataset,
                "axis": axis,
                "metric": metric,
                "n_severities": int(len(ordered)),
                "breaks": first is not None,
                "breakpoint_severity_tag": None if first is None else first["severity_tag"],
                "breakpoint_severity": None
                if first is None
                else severity_by_tag.get((axis, first["severity_tag"])),
                "difference_at_breakpoint": None if first is None else first["mean_difference"],
                "ci_at_breakpoint": None
                if first is None
                else f"[{first['ci_low']:.4f}, {first['ci_high']:.4f}]",
                "worst_difference": float(ordered["mean_difference"].min()),
                "worst_severity_tag": ordered.loc[
                    ordered["mean_difference"].idxmin(), "severity_tag"
                ],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Hyperparameter influence on the breakpoint
# --------------------------------------------------------------------------
def _ordered_severity_tags(metrics_frame: pd.DataFrame, axis: str) -> list[str]:
    """Severity tags for one axis, ordered anchor-first, most-degraded-last."""
    spec = get_axis(axis)
    sub = metrics_frame.loc[metrics_frame["axis"] == axis].drop_duplicates("severity_tag")

    def order(severity: Any) -> float:
        if severity is None or (isinstance(severity, str) and severity == "full"):
            return spec.severity_order(None)
        try:
            return spec.severity_order(float(severity))
        except (TypeError, ValueError):
            return float("inf")

    pairs = [(row["severity_tag"], order(row["severity"])) for _, row in sub.iterrows()]
    return [tag for tag, _ in sorted(pairs, key=lambda pair: pair[1])]


def variant_families(per_cell: dict[tuple, dict[str, Any]]) -> dict[str, list[str]]:
    """Base models that were fitted under more than one hyperparameter setting."""
    all_models = sorted({model for result in per_cell.values() for model in result["models"]})
    families: dict[str, list[str]] = {}
    for base in sorted({base_model(model) for model in all_models}):
        members = sorted(model for model in all_models if base_model(model) == base)
        if len(members) > 1:
            families[base] = members
    return families


def _hparam_value_map(metrics_frame: pd.DataFrame) -> dict[str, Any]:
    """Series id -> the swept hyperparameter value it was fitted with, if recorded."""
    if "hparam_value" not in metrics_frame.columns:
        return {}
    mapping: dict[str, Any] = {}
    for model, group in metrics_frame.dropna(subset=["model"]).groupby("model"):
        values = group["hparam_value"].dropna().unique()
        if len(values):
            mapping[str(model)] = values[0]
    return mapping


def hyperparameter_influence(
    per_cell: dict[tuple, dict[str, Any]],
    metrics_frame: pd.DataFrame,
    families: dict[str, list[str]],
    metric: str,
    reference: str,
    level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Locate each hyperparameter variant's breakpoint and its shift vs the default.

    For every base model that was swept, each variant is compared -- paired,
    exactly like the headline H3 test -- against the strongest model that is *not*
    a sibling variant, and its breakpoint is found. The variant sharing the plain
    base name is the reference default; the ``robustness_index`` (how many
    severity steps in the variant survives before its advantage turns
    significantly negative, or the full grid length if it never does) is compared
    against that default to report how far the hyperparameter moved the breakpoint.
    """
    value_map = _hparam_value_map(metrics_frame)
    per_variant_rows: list[dict[str, Any]] = []
    for base, members in families.items():
        for variant_id in members:
            paired = paired_comparisons(per_cell, metric, variant_id, level)
            breakpoints = find_breakpoints(paired, metrics_frame)
            for _, row in breakpoints.iterrows():
                per_variant_rows.append(
                    {
                        "dataset": row["dataset"],
                        "axis": row["axis"],
                        "metric": metric,
                        "base_model": base,
                        "model": variant_id,
                        "hparam_tag": hparam_tag(variant_id),
                        "hparam_value": value_map.get(variant_id),
                        "is_default": hparam_tag(variant_id) == "default",
                        "breaks": bool(row["breaks"]),
                        "breakpoint_severity_tag": row["breakpoint_severity_tag"],
                        "breakpoint_severity": row["breakpoint_severity"],
                        "difference_at_breakpoint": row["difference_at_breakpoint"],
                        "worst_difference": row["worst_difference"],
                        "worst_severity_tag": row["worst_severity_tag"],
                    }
                )
    per_variant = pd.DataFrame(per_variant_rows)
    if per_variant.empty:
        return per_variant, pd.DataFrame()

    # robustness_index: position of the breakpoint along the ordered severity
    # grid. A larger index means the advantage survived more degradation.
    ordered_by_axis = {
        axis: _ordered_severity_tags(metrics_frame, axis)
        for axis in per_variant["axis"].unique()
    }

    def robustness_index(axis: str, breaks: bool, tag: Optional[str]) -> int:
        order = ordered_by_axis.get(axis, [])
        if not breaks or tag is None or tag not in order:
            return len(order)  # never broke within the grid: maximally robust
        return int(order.index(tag))

    per_variant["n_severities"] = per_variant["axis"].map(
        lambda axis: len(ordered_by_axis.get(axis, []))
    )
    per_variant["robustness_index"] = per_variant.apply(
        lambda row: robustness_index(
            row["axis"], row["breaks"], row["breakpoint_severity_tag"]
        ),
        axis=1,
    )

    shift_rows: list[dict[str, Any]] = []
    for (dataset, axis, base), group in per_variant.groupby(
        ["dataset", "axis", "base_model"]
    ):
        defaults = group.loc[group["is_default"]]
        if defaults.empty:
            continue
        default_index = int(defaults.iloc[0]["robustness_index"])
        default_tag = defaults.iloc[0]["breakpoint_severity_tag"]
        for _, row in group.iterrows():
            if row["is_default"]:
                continue
            shift = int(row["robustness_index"]) - default_index
            shift_rows.append(
                {
                    "dataset": dataset,
                    "axis": axis,
                    "base_model": base,
                    "hparam_tag": row["hparam_tag"],
                    "hparam_value": row["hparam_value"],
                    "variant_breaks": bool(row["breaks"]),
                    "variant_breakpoint": row["breakpoint_severity_tag"],
                    "default_breaks": bool(defaults.iloc[0]["breaks"]),
                    "default_breakpoint": default_tag,
                    # Steps the breakpoint moved: positive => this value survives
                    # deeper into the degradation than the default (more robust);
                    # negative => it breaks earlier (more fragile); 0 => unchanged.
                    "breakpoint_shift_steps": shift,
                    "more_robust_than_default": shift > 0,
                    "more_fragile_than_default": shift < 0,
                }
            )
    return per_variant, pd.DataFrame(shift_rows)


# --------------------------------------------------------------------------
# PEHE and the H2 agreement test
# --------------------------------------------------------------------------
def pehe_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if "pehe" not in metrics.columns:
        return pd.DataFrame()
    usable = metrics.loc[(metrics["status"] == "ok") & metrics["pehe"].notna()]
    if usable.empty:
        return pd.DataFrame()
    aggregations = {"pehe": ["mean", "std", "count"], "qini_auc_normalized": ["mean"]}
    if "abs_ate_error" in usable.columns:
        aggregations["abs_ate_error"] = ["mean"]
    table = usable.groupby(["dataset", "axis", "severity_tag", "model"]).agg(aggregations)
    table.columns = ["_".join(part for part in name if part) for name in table.columns]
    return table.reset_index()


def spearman_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """H2 inside each regime: does effect accuracy predict ranking quality?"""
    if "pehe" not in metrics.columns:
        return pd.DataFrame()
    usable = metrics.loc[(metrics["status"] == "ok") & metrics["pehe"].notna()]
    if usable.empty:
        return pd.DataFrame()

    rows = []
    averaged = (
        usable.groupby(["dataset", "axis", "severity_tag", "model"])[
            ["pehe", "qini_auc_normalized"]
        ]
        .mean()
        .reset_index()
    )
    for (dataset, axis, severity_tag), group in averaged.groupby(
        ["dataset", "axis", "severity_tag"]
    ):
        agreement = ranking_agreement(
            dict(zip(group["model"], group["pehe"])),
            dict(zip(group["model"], group["qini_auc_normalized"])),
        )
        rows.append(
            {
                "dataset": dataset,
                "axis": axis,
                "severity_tag": severity_tag,
                "n_methods": agreement["n_methods"],
                "spearman_rho": agreement["spearman_rho"],
                "p_value": agreement["p_value"],
                "value_correlation": agreement["value_correlation"],
                "pehe_order": "|".join(agreement.get("pehe_order", [])),
                "qini_order": "|".join(agreement.get("qini_order", [])),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def write_figures(pooled: pd.DataFrame, metric: str, output_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping figures.")
        return []

    written = []
    for (dataset, axis), group in pooled.groupby(["dataset", "axis"]):
        if axis not in AXES:
            continue
        figure, ax = plt.subplots(figsize=(7, 4.2))
        for model, series in group.groupby("model"):
            series = series.sort_values("severity_tag")
            x = np.arange(len(series))
            ax.plot(x, series["mean"], marker="o", label=model)
            ax.fill_between(x, series["ci_low"], series["ci_high"], alpha=0.15)
            ax.set_xticks(x)
            ax.set_xticklabels(series["severity_tag"], rotation=30, ha="right", fontsize=7)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(f"{dataset} - {axis}")
        ax.set_ylabel(metric)
        ax.legend(fontsize=7)
        figure.tight_layout()
        path = output_dir / f"figure_{dataset}_{axis}_{metric}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path.name)
    return written


def write_hparam_figures(
    pooled: pd.DataFrame,
    families: dict[str, list[str]],
    metric: str,
    output_dir: Path,
) -> list[str]:
    """One severity-vs-metric figure per (dataset, axis, swept model), one line
    per hyperparameter variant. This is the picture that shows, directly, whether
    a hyperparameter moves the curve or leaves it on top of itself."""
    if not families:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping hyperparameter figures.")
        return []

    members_by_base = {base: set(members) for base, members in families.items()}
    written = []
    for (dataset, axis), group in pooled.groupby(["dataset", "axis"]):
        if axis not in AXES:
            continue
        for base, members in members_by_base.items():
            family = group.loc[group["model"].isin(members)]
            if family["model"].nunique() < 2:
                continue
            figure, ax = plt.subplots(figsize=(7, 4.2))
            for model, series in family.groupby("model"):
                series = series.sort_values("severity_tag")
                x = np.arange(len(series))
                label = "default" if base_model(model) == model else hparam_tag(model)
                ax.plot(x, series["mean"], marker="o", label=label)
                ax.fill_between(x, series["ci_low"], series["ci_high"], alpha=0.15)
                ax.set_xticks(x)
                ax.set_xticklabels(series["severity_tag"], rotation=30, ha="right", fontsize=7)
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
            ax.set_title(f"{dataset} - {axis} - {base} (hyperparameter sweep)")
            ax.set_ylabel(metric)
            ax.legend(fontsize=7, title="variant")
            figure.tight_layout()
            path = output_dir / f"hparam_{dataset}_{axis}_{base}_{metric}.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            written.append(path.name)
    return written


def _hparam_markdown(
    shift_summary: pd.DataFrame, per_variant: pd.DataFrame
) -> list[str]:
    """The 'which hyperparameters matter' section of the regime map."""
    lines = ["", "## Hyperparameter influence on the breakpoint", ""]
    if shift_summary.empty:
        lines.append(
            "_No hyperparameter variants in this sweep; pass e.g. "
            "`--causalpfn-context-variants 1024,4096` to populate this section._"
        )
        return lines

    lines.append(
        "`breakpoint_shift_steps` counts severity grid steps: positive means the "
        "value survived deeper into the degradation than the model's default "
        "(more robust), negative means it broke earlier (more fragile), 0 means "
        "the breakpoint did not move."
    )
    lines.append("")
    lines.append(
        "| dataset | axis | model | hyperparameter | default | variant break | shift (steps) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for _, row in shift_summary.sort_values(
        ["dataset", "axis", "base_model", "hparam_value"]
    ).iterrows():
        variant_break = row["variant_breakpoint"] or ("no break" if not row["variant_breaks"] else "-")
        default_break = row["default_breakpoint"] or ("no break" if not row["default_breaks"] else "-")
        lines.append(
            f"| {row['dataset']} | {row['axis']} | {row['base_model']} | "
            f"{row['hparam_tag']} ({row['hparam_value']}) | {default_break} | "
            f"{variant_break} | {row['breakpoint_shift_steps']:+d} |"
        )

    # A compact verdict per (model, hyperparameter): the largest absolute shift it
    # produced across every axis and value tells the reader whether it is worth
    # tuning at all.
    lines += ["", "### Verdict: which hyperparameters are critical", ""]
    verdict = (
        shift_summary.assign(abs_shift=shift_summary["breakpoint_shift_steps"].abs())
        .groupby("base_model")["abs_shift"]
        .max()
        .sort_values(ascending=False)
    )
    for base, max_shift in verdict.items():
        if max_shift == 0:
            note = "not critical: no value moved any breakpoint."
        elif max_shift == 1:
            note = "marginal: a single grid step at most."
        else:
            note = f"critical: shifts the breakpoint by up to {int(max_shift)} grid steps."
        lines.append(f"- **{base}** — {note}")
    return lines


def write_regime_map(
    output_dir: Path,
    breakpoints: pd.DataFrame,
    spearman: pd.DataFrame,
    pooled: pd.DataFrame,
    metric: str,
    reference: str,
    shift_summary: Optional[pd.DataFrame] = None,
    per_variant: Optional[pd.DataFrame] = None,
) -> None:
    lines = [
        "# Regime breakdown map",
        "",
        f"Primary metric: `{metric}`. Reference model: `{reference}`.",
        "",
        "Bootstrap intervals resample evaluation rows with replacement, paired "
        "across models within a seed and pooled across seeds.",
        "",
        "## Where the advantage breaks",
        "",
    ]
    if breakpoints.empty:
        lines.append("_No paired comparisons available._")
    else:
        lines.append(
            "| dataset | axis | metric | breaks | breakpoint | difference | interval | worst |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, row in breakpoints.sort_values(["dataset", "axis", "metric"]).iterrows():
            difference = row["difference_at_breakpoint"]
            difference_text = "-" if difference is None else f"{float(difference):.4f}"
            lines.append(
                f"| {row['dataset']} | {row['axis']} | {row['metric']} | "
                f"{'yes' if row['breaks'] else 'no'} | "
                f"{row['breakpoint_severity_tag'] or '-'} | {difference_text} | "
                f"{row['ci_at_breakpoint'] or '-'} | "
                f"{row['worst_difference']:.4f} at {row['worst_severity_tag']} |"
            )

    lines += ["", "## PEHE / Qini ranking agreement (H2)", ""]
    if spearman.empty:
        lines.append("_No ground-truth data in this sweep; run a `semisynth` dataset._")
    else:
        lines.append("| dataset | axis | severity | n methods | Spearman rho | p |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in spearman.sort_values(["dataset", "axis", "severity_tag"]).iterrows():
            lines.append(
                f"| {row['dataset']} | {row['axis']} | {row['severity_tag']} | "
                f"{row['n_methods']} | {row['spearman_rho']:.3f} | {row['p_value']:.3f} |"
            )

    if shift_summary is not None:
        lines += _hparam_markdown(shift_summary, per_variant if per_variant is not None else pd.DataFrame())

    lines += ["", "## Cells analysed", "", f"{len(pooled)} (dataset, axis, severity, model) rows.", ""]
    (output_dir / "regime_map.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (args.results_root / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = [item.strip() for item in args.metrics.split(",") if item.strip()]
    max_rows = None if args.bootstrap_max_rows == 0 else args.bootstrap_max_rows

    metrics = load_metrics(args.results_root)
    metrics.to_csv(output_dir / "all_metrics.csv", index=False)

    per_metric_rows, per_cell = bootstrap_all_cells(
        args.results_root,
        metrics=metric_names,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed,
        max_rows=max_rows,
        level=args.level,
    )
    per_metric_rows.to_csv(output_dir / "bootstrap_per_seed.csv", index=False)

    primary = metric_names[0]
    all_pooled, all_paired = [], []
    for metric in metric_names:
        pooled = pool_across_seeds(per_cell, metric, args.level)
        paired = paired_comparisons(per_cell, metric, args.reference_model, args.level)
        all_pooled.append(pooled)
        all_paired.append(paired)
        for axis in sorted(pooled["axis"].unique()) if not pooled.empty else []:
            pooled.loc[pooled["axis"] == axis].to_csv(
                output_dir / f"qini_bootstrap_{axis}_{metric}.csv", index=False
            )
            if not paired.empty:
                paired.loc[paired["axis"] == axis].to_csv(
                    output_dir / f"paired_{axis}_{metric}.csv", index=False
                )

    pooled = pd.concat(all_pooled, ignore_index=True) if all_pooled else pd.DataFrame()
    paired = pd.concat(all_paired, ignore_index=True) if all_paired else pd.DataFrame()
    breakpoints = find_breakpoints(paired, metrics)
    if not breakpoints.empty:
        for axis in sorted(breakpoints["axis"].unique()):
            breakpoints.loc[breakpoints["axis"] == axis].to_csv(
                output_dir / f"breakpoint_{axis}.csv", index=False
            )

    # Hyperparameter influence: per swept model, how far each value moved the
    # breakpoint relative to the model's default. Only populated when the sweep
    # actually fitted more than one value for some model.
    families = variant_families(per_cell)
    per_variant, shift_summary = hyperparameter_influence(
        per_cell, metrics, families, primary, args.reference_model, args.level
    )
    if not per_variant.empty:
        per_variant.to_csv(output_dir / "hparam_breakpoints.csv", index=False)
        for axis in sorted(per_variant["axis"].unique()):
            per_variant.loc[per_variant["axis"] == axis].to_csv(
                output_dir / f"breakpoint_hparam_{axis}.csv", index=False
            )
    if not shift_summary.empty:
        shift_summary.to_csv(output_dir / "hparam_breakpoint_shift.csv", index=False)

    pehe_rows = pehe_table(metrics)
    spearman_rows = spearman_table(metrics)
    for frame, prefix in ((pehe_rows, "pehe"), (spearman_rows, "spearman")):
        if frame.empty:
            continue
        for axis in sorted(frame["axis"].unique()):
            frame.loc[frame["axis"] == axis].to_csv(
                output_dir / f"{prefix}_{axis}.csv", index=False
            )

    figures = []
    if not args.no_figures and not pooled.empty:
        primary_pooled = pooled.loc[pooled["metric"] == primary]
        figures = write_figures(primary_pooled, primary, output_dir)
        figures += write_hparam_figures(primary_pooled, families, primary, output_dir)

    write_regime_map(
        output_dir,
        breakpoints,
        spearman_rows,
        pooled,
        primary,
        args.reference_model,
        shift_summary,
        per_variant,
    )
    (output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                **{k: str(v) for k, v in vars(args).items()},
                "n_cells_bootstrapped": len(per_cell),
                "figures": figures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Analysis written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
