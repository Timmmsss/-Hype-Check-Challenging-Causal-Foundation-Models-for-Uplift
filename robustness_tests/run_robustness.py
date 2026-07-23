#!/usr/bin/env python
"""Sweep the three robustness axes (project goal G2).

For every (dataset, axis, severity, seed) cell this fits each requested model on
the degraded training data and evaluates it on the held-out split, writing
results in the same layout as ``baseline_benchmark/run_baselines.py`` plus the
axis columns the analysis needs.

Two protocol rules are enforced here rather than left to the operator:

* Hyperparameters are **frozen** across severities (``--frozen-params``). Tuning
  per severity would confound the axis with the search budget.
* A cell that cannot be estimated -- an arm lost, too few converters for the
  Q-Learner -- is recorded with a non-``ok`` status and the sweep continues,
  because "not estimable at this severity" is itself a finding.

Examples
--------
Preview the full matrix without running anything::

    python run_robustness.py --datasets retailhero --axes all --dry-run

One axis, ten seeds, frozen hyperparameters::

    python run_robustness.py --datasets hillstrom --axes conversion \\
        --seeds 0,1,2,3,4,5,6,7,8,9 --frozen-params <tuning>/best_params.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from joblib import dump

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robustness_tests.axes import (  # noqa: E402
    AXES,
    DEGRADE_CHOICES,
    AxisSpec,
    default_models,
    get_axis,
    parse_grid,
)
from robustness_tests.pehe import abs_ate_error, pehe  # noqa: E402
from robustness_tests.prepare import DegenerateRegimeError, prepare_resampled_data  # noqa: E402
from robustness_tests.registry import (  # noqa: E402
    available_robustness_models,
    make_robustness_model,
    supports_pehe,
)
from robustness_tests.reuse import IncompatibleRunError, find_reusable_run, ingest_run  # noqa: E402
from robustness_tests.semisynth import SemiSyntheticSpec, prepare_semisynthetic_data  # noqa: E402

from baseline_benchmark.data import DATASET_SPECS  # noqa: E402
from baseline_benchmark.metrics import evaluate_uplift  # noqa: E402

SEMISYNTH_PREFIX = "semisynth"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep the scale, control-share and conversion robustness axes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        default="retailhero",
        help=(
            "Comma-separated cleaned dataset names, and/or 'semisynth' for the "
            "binary semi-synthetic generator with known ground-truth CATE."
        ),
    )
    parser.add_argument("--axes", default="all", help=f"Comma-separated, or 'all'. {sorted(AXES)}")
    parser.add_argument(
        "--severities",
        default=None,
        help="Comma-separated severity grid. Only valid with a single --axes value.",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated. Defaults to the core baselines plus the axis's specialist.",
    )
    parser.add_argument("--outcome", default=None)
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "data_A_cleaned",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results_robustness",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Global pre-split cap, independent of the scale axis. 0 means all rows.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--evaluation-split", choices=["validation", "test"], default="test")
    parser.add_argument(
        "--degrade-splits",
        choices=list(DEGRADE_CHOICES),
        default=None,
        help="Override the per-axis default (scale/control_share: train+val; conversion: all).",
    )
    parser.add_argument(
        "--conversion-risk",
        default="fitted",
        help="'fitted' (train-only P(Y=1|C) model) or 'natural:<raw column>'.",
    )
    parser.add_argument("--conversion-risk-sign", type=float, default=1.0)
    parser.add_argument(
        "--equalize-rows",
        type=int,
        default=0,
        help=(
            "Trim every severity to this many training rows (splits scaled in "
            "proportion) after resampling. Strongly recommended on the conversion "
            "and control-share axes: a rule that thins hard also shrinks the "
            "sample, which would otherwise confound the axis with the scale axis. "
            "0 disables."
        ),
    )
    parser.add_argument(
        "--frozen-params",
        type=Path,
        default=None,
        help="best_params.json from the seed-42 tuning batch; applied at every severity.",
    )
    parser.add_argument(
        "--reuse-existing",
        type=Path,
        default=None,
        help=(
            "Ingest completed baseline runs for the undegraded anchor of each axis "
            "instead of refitting them (the expensive full-table CausalPFN runs). "
            "The earlier run's protocol must match, or the cell is refused."
        ),
    )
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument(
        "--allow-confounding",
        action="store_true",
        help="Permit a covariate-dependent P*(T|C). This makes Qini non-causal.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the run matrix and stop.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cells that already have a completed metrics.csv under --output-root.",
    )

    # Model hyperparameters, mirroring run_baselines.py so a robustness run can
    # be compared against the existing baseline runs.
    parser.add_argument("--tree-max-iter", type=int, default=150)
    parser.add_argument("--tree-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--tree-learning-rate", type=float, default=0.05)
    parser.add_argument("--dr-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--neural-learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--causalpfn-model-path", default="vdblm/causalpfn")
    parser.add_argument("--causalpfn-cache-dir", type=Path, default=None)
    parser.add_argument("--causalpfn-max-context-length", type=int, default=4096)
    parser.add_argument("--causalpfn-max-query-length", type=int, default=4096)
    parser.add_argument("--causalpfn-num-neighbours", type=int, default=1024)
    parser.add_argument("--causalpfn-calibrate", action="store_true")
    parser.add_argument("--causalpfn-verbose", action="store_true")
    parser.add_argument("--q-min-converters", type=int, default=50)
    parser.add_argument("--gp-max-control", type=int, default=2000)
    parser.add_argument("--gp-max-treated", type=int, default=500)
    parser.add_argument("--gp-n-components", type=int, default=20)

    # Semi-synthetic generator.
    parser.add_argument("--semisynth-covariates", default="gaussian:20000x10")
    parser.add_argument("--semisynth-base-rate", type=float, default=0.05)
    parser.add_argument("--semisynth-treatment-rate", type=float, default=0.5)
    parser.add_argument("--semisynth-ate", type=float, default=0.01)
    parser.add_argument("--semisynth-heterogeneity", type=float, default=1.0)
    parser.add_argument("--semisynth-signal-features", type=int, default=5)
    return parser.parse_args(argv)


def _parse_list(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _resolve_axes(value: str) -> list[AxisSpec]:
    names = _parse_list(value)
    if names == ["all"]:
        names = list(AXES)
    return [get_axis(name) for name in names]


def _resolve_models(args: argparse.Namespace, axis: AxisSpec) -> list[str]:
    names = _parse_list(args.models) if args.models else default_models(axis)
    unknown = sorted(set(names) - set(available_robustness_models()))
    if unknown:
        raise ValueError(
            f"Unknown models {unknown}; available={available_robustness_models()}"
        )
    return names


def _load_frozen(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    """Read ``best_params.json`` as written by the parallel tuning batch."""
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frozen: dict[str, dict[str, Any]] = {}
    for model, entry in payload.items():
        params = entry.get("params") if isinstance(entry, dict) else None
        if isinstance(params, dict):
            frozen[str(model).lower()] = dict(params)
    return frozen


def _model_kwargs(args: argparse.Namespace, model_name: str, seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "max_iter": args.tree_max_iter,
        "max_leaf_nodes": args.tree_max_leaf_nodes,
        "learning_rate": (
            args.neural_learning_rate if model_name == "dragonnet" else args.tree_learning_rate
        ),
        "n_folds": args.dr_folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "patience": args.patience,
        "device": args.device,
        "model_path": args.causalpfn_model_path,
        "cache_dir": args.causalpfn_cache_dir,
        "max_context_length": args.causalpfn_max_context_length,
        "max_query_length": args.causalpfn_max_query_length,
        "num_neighbours": args.causalpfn_num_neighbours,
        "calibrate": args.causalpfn_calibrate,
        "verbose": args.causalpfn_verbose,
        "min_converters": args.q_min_converters,
        "max_control": args.gp_max_control,
        "max_treated": args.gp_max_treated,
        "n_components": args.gp_n_components,
    }


def _evaluation_arrays(resampled, evaluation_split: str):
    data = resampled.prepared
    if evaluation_split == "validation":
        return data.X_val, data.id_val, data.t_val, data.y_val, resampled.extras.get(
            "validation", {}
        ).get("tau_true")
    return data.X_test, data.id_test, data.t_test, data.y_test, resampled.extras.get(
        "test", {}
    ).get("tau_true")


def _split_stats(t: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Same shape as ``run_baselines.py::_split_stats``, for manifest comparability."""
    return {
        "n": int(len(y)),
        "treatment_rate": float(np.mean(t)),
        "outcome_rate": float(np.mean(y)),
        "n_control": int(np.sum(t == 0)),
        "n_treated": int(np.sum(t == 1)),
        "events_control": int(np.sum(y[t == 0])),
        "events_treated": int(np.sum(y[t == 1])),
    }


def _cell_directory(
    output_root: Path, dataset: str, axis: AxisSpec, severity_tag: str, seed: int, stamp: str
) -> Path:
    return output_root / dataset / axis.name / severity_tag / f"seed_{seed}_{stamp}"


def _already_done(output_root: Path, dataset: str, axis: AxisSpec, severity_tag: str, seed: int) -> bool:
    parent = output_root / dataset / axis.name / severity_tag
    if not parent.exists():
        return False
    return any(
        (candidate / "metrics.csv").exists()
        for candidate in parent.glob(f"seed_{seed}_*")
        if candidate.is_dir()
    )


def _prepare_cell(args: argparse.Namespace, dataset: str, axis: AxisSpec, severity: Any, seed: int):
    shared = {
        "axis": axis.name,
        "severity": severity,
        "seed": seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "degrade_splits": args.degrade_splits,
        "conversion_risk": args.conversion_risk,
        "conversion_risk_sign": args.conversion_risk_sign,
        "run_diagnostics": not args.no_diagnostics,
        "allow_confounding": args.allow_confounding,
        "equalize_rows": None if args.equalize_rows == 0 else args.equalize_rows,
    }
    max_rows = None if args.max_rows == 0 else args.max_rows
    if dataset.startswith(SEMISYNTH_PREFIX):
        spec = SemiSyntheticSpec(
            covariate_source=args.semisynth_covariates,
            base_rate=args.semisynth_base_rate,
            treatment_rate=args.semisynth_treatment_rate,
            ate=args.semisynth_ate,
            heterogeneity=args.semisynth_heterogeneity,
            n_signal_features=args.semisynth_signal_features,
            seed=42,
        )
        return prepare_semisynthetic_data(
            spec,
            cleaned_root=args.cleaned_root,
            max_rows=max_rows,
            dataset_label=dataset,
            **shared,
        )
    return prepare_resampled_data(
        cleaned_root=args.cleaned_root,
        dataset=dataset,
        outcome=args.outcome,
        max_rows=max_rows,
        **shared,
    )


def _reuse_requirements(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    """Protocol fields an existing run must match to be reusable."""
    return {
        "max_rows_resolved": None if args.max_rows == 0 else args.max_rows,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "evaluation_split": args.evaluation_split,
        "causalpfn_max_context_length": args.causalpfn_max_context_length,
        "causalpfn_max_query_length": args.causalpfn_max_query_length,
        "causalpfn_num_neighbours": args.causalpfn_num_neighbours,
    }


def _try_reuse(
    args: argparse.Namespace,
    dataset: str,
    axis: AxisSpec,
    severity: Any,
    seed: int,
    models: list[str],
    base_row: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest a completed run for an anchor cell, if one matches."""
    if args.reuse_existing is None or not axis.is_anchor(severity):
        return pd.DataFrame(), pd.DataFrame()
    if dataset.startswith(SEMISYNTH_PREFIX):
        return pd.DataFrame(), pd.DataFrame()
    try:
        run_dir = find_reusable_run(
            args.reuse_existing, dataset, seed, _reuse_requirements(args, dataset)
        )
    except IncompatibleRunError as exc:
        print(f"REUSE REFUSED {dataset}/seed_{seed}: {exc}", flush=True)
        return pd.DataFrame(), pd.DataFrame()
    if run_dir is None:
        return pd.DataFrame(), pd.DataFrame()
    metrics, predictions = ingest_run(run_dir, models, base_row)
    if not metrics.empty:
        print(
            f"REUSE {dataset}/{axis.name}/{base_row['severity_tag']}/seed_{seed}: "
            f"{sorted(metrics['model'].unique())} from {run_dir}",
            flush=True,
        )
    return metrics, predictions


def run_cell(
    args: argparse.Namespace,
    dataset: str,
    axis: AxisSpec,
    severity: Any,
    seed: int,
    models: list[str],
    frozen: dict[str, dict[str, Any]],
    stamp: str,
) -> list[dict[str, Any]]:
    """Run every model for one cell and write its artifacts. Never raises."""
    severity_tag = axis.tag(severity)
    output_dir = _cell_directory(args.output_root, dataset, axis, severity_tag, seed, stamp)
    base_row = {
        "dataset": dataset,
        "axis": axis.name,
        "severity": "full" if severity is None else severity,
        "severity_tag": severity_tag,
        "severity_kind": axis.severity_kind,
        "seed": seed,
        "evaluation_split": args.evaluation_split,
        "degrade_splits": args.degrade_splits or axis.default_degrade_splits,
    }

    reused_metrics, reused_predictions = _try_reuse(
        args, dataset, axis, severity, seed, models, base_row
    )
    reused_models = (
        set(reused_metrics["model"].astype(str)) if not reused_metrics.empty else set()
    )
    models_to_fit = [name for name in models if name not in reused_models]
    if reused_models and not models_to_fit:
        # Every requested model was already evaluated on this exact split, so
        # the cell needs no data preparation at all. This is the case that saves
        # the full-table CausalPFN hours.
        output_dir.mkdir(parents=True, exist_ok=True)
        reused_metrics.to_csv(output_dir / "metrics.csv", index=False)
        reused_predictions.to_parquet(output_dir / "predictions.parquet", index=False)
        return reused_metrics.to_dict("records")

    try:
        resampled = _prepare_cell(args, dataset, axis, severity, seed)
    except (DegenerateRegimeError, ValueError) as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {**base_row, "outcome": args.outcome or "", "model": model,
             "status": "degenerate_regime", "error": str(exc)}
            for model in models
        ]
        pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
        (output_dir / "resample_manifest.json").write_text(
            json.dumps({"error": str(exc), **base_row}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"SKIP {dataset}/{axis.name}/{severity_tag}/seed_{seed}: {exc}", flush=True)
        return rows

    data = resampled.prepared
    output_dir.mkdir(parents=True, exist_ok=True)
    data.split_table.to_parquet(output_dir / "splits.parquet", index=False)
    dump(data.preprocessor, output_dir / "preprocessor.joblib")
    pd.DataFrame({"feature_name": data.feature_names}).to_csv(
        output_dir / "transformed_features.csv", index=False
    )
    (output_dir / "resample_manifest.json").write_text(
        json.dumps(
            {
                **base_row,
                "resample_records": resampled.resample_records,
                "diagnostics": resampled.diagnostics,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (output_dir / "data_manifest.json").write_text(
        json.dumps(
            {
                **base_row,
                "outcome": data.outcome,
                "group_safe_split": data.group_safe,
                "n_transformed_features": len(data.feature_names),
                "transformed_feature_names": data.feature_names,
                "train": _split_stats(data.t_train, data.y_train),
                "validation": _split_stats(data.t_val, data.y_val),
                "test": _split_stats(data.t_test, data.y_test),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    X_eval, id_eval, t_eval, y_eval, tau_eval = _evaluation_arrays(resampled, args.evaluation_split)
    realized = resampled.diagnostics.get("realized", {})
    shared_row = {
        **base_row,
        "outcome": data.outcome,
        "n_train": int(len(data.y_train)),
        "n_validation": int(len(data.y_val)),
        "n_test": int(len(data.y_test)),
        "n_evaluation": int(len(y_eval)),
        "n_features": int(data.X_train.shape[1]),
        "degenerate": bool(resampled.diagnostics.get("degenerate", False)),
        "equalization_reached": bool(resampled.diagnostics.get("equalization_reached", True)),
        "train_treatment_rate": realized.get("train", {}).get("realized_treatment_rate"),
        "train_conversion_rate": realized.get("train", {}).get("realized_conversion_rate"),
        "eval_treatment_rate": realized.get(
            "test" if args.evaluation_split == "test" else "validation", {}
        ).get("realized_treatment_rate"),
        "eval_conversion_rate": realized.get(
            "test" if args.evaluation_split == "test" else "validation", {}
        ).get("realized_conversion_rate"),
        "has_ground_truth": tau_eval is not None,
    }

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    if not reused_predictions.empty:
        # Reused rows may only be mixed in when they scored the very same units.
        reused_ids = np.sort(reused_predictions["epk_id"].unique())
        if not np.array_equal(reused_ids, np.sort(np.asarray(id_eval))):
            raise RuntimeError(
                f"Reused predictions for {dataset}/seed_{seed} cover different evaluation "
                "rows than the freshly prepared split; refusing to combine them."
            )
        metrics_rows.extend(reused_metrics.to_dict("records"))
        prediction_frames.append(reused_predictions)

    for model_name in models_to_fit:
        kwargs = _model_kwargs(args, model_name, seed)
        kwargs.update(frozen.get(model_name, {}))
        row: dict[str, Any] = {**shared_row, "model": model_name}
        try:
            model = make_robustness_model(model_name, **kwargs)
            fit_start = time.perf_counter()
            model.fit(
                data.X_train, data.t_train, data.y_train,
                data.X_val, data.t_val, data.y_val,
            )
            row["fit_seconds"] = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            cate = np.asarray(model.predict_cate(X_eval), dtype=float)
            row["predict_seconds"] = time.perf_counter() - predict_start
            if cate.shape != y_eval.shape or not np.isfinite(cate).all():
                raise RuntimeError(f"{model_name} returned invalid CATE predictions")

            row.update(evaluate_uplift(y_eval, cate, t_eval))
            if tau_eval is not None and supports_pehe(model_name):
                row["pehe"] = pehe(tau_eval, cate)
                row["abs_ate_error"] = abs_ate_error(tau_eval, cate)
            elif tau_eval is not None:
                # Ratio-scale scores are not comparable to a difference-scale
                # ground truth; ranking metrics above still apply.
                row["pehe"] = float("nan")
                row["abs_ate_error"] = float("nan")
            row["status"] = "ok"

            frame = pd.DataFrame(
                {
                    "epk_id": id_eval,
                    "dataset": dataset,
                    "outcome": data.outcome,
                    "axis": axis.name,
                    "severity_tag": severity_tag,
                    "model": model_name,
                    "seed": seed,
                    "evaluation_split": args.evaluation_split,
                    "T": t_eval,
                    "Y": y_eval,
                    "cate_pred": cate,
                }
            )
            if tau_eval is not None:
                frame["tau_true"] = tau_eval
            prediction_frames.append(frame)
        except Exception as exc:  # noqa: BLE001 - one model must not kill the sweep
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"FAIL {dataset}/{axis.name}/{severity_tag}/seed_{seed}/{model_name}: {exc}",
                flush=True,
            )
            if args.causalpfn_verbose:
                traceback.print_exc()
        metrics_rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "error"}, default=str), flush=True)

    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_parquet(
            output_dir / "predictions.parquet", index=False
        )
    return metrics_rows


def build_matrix(args: argparse.Namespace) -> list[tuple[str, AxisSpec, Any, int, list[str]]]:
    datasets = _parse_list(args.datasets)
    unknown = [
        name
        for name in datasets
        if name not in DATASET_SPECS and not name.startswith(SEMISYNTH_PREFIX)
    ]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s) {unknown}; choose from {sorted(DATASET_SPECS)} "
            f"or a name starting with {SEMISYNTH_PREFIX!r}"
        )
    axes = _resolve_axes(args.axes)
    if args.severities and len(axes) != 1:
        raise ValueError("--severities requires exactly one value in --axes")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")

    matrix = []
    for dataset in datasets:
        for axis in axes:
            models = _resolve_models(args, axis)
            for severity in parse_grid(axis, args.severities):
                for seed in seeds:
                    matrix.append((dataset, axis, severity, seed, models))
    return matrix


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    matrix = build_matrix(args)
    frozen = _load_frozen(args.frozen_params)

    if args.dry_run:
        print(f"{len(matrix)} cells:")
        for dataset, axis, severity, seed, models in matrix:
            print(
                f"  {dataset:<16} {axis.name:<14} {axis.tag(severity):<26} "
                f"seed={seed:<3} models={','.join(models)}"
            )
        n_fits = sum(len(models) for *_, models in matrix)
        print(f"\n{len(matrix)} cells, {n_fits} model fits.")
        if frozen:
            print(f"Frozen hyperparameters loaded for: {sorted(frozen)}")
        else:
            print("No --frozen-params given; command-line defaults apply at every severity.")
        thinning_axes = {axis.name for _, axis, *_ in matrix} & {"conversion", "control_share"}
        if thinning_axes and args.equalize_rows == 0:
            print(
                f"\nWARNING: {sorted(thinning_axes)} thin the sample as severity rises, so "
                "training size varies along the axis and is confounded with the scale axis. "
                "Pass --equalize-rows N (N at or below the harshest severity's yield) to "
                "hold sample size fixed."
            )
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for index, (dataset, axis, severity, seed, models) in enumerate(matrix, start=1):
        tag = axis.tag(severity)
        if args.skip_existing and _already_done(args.output_root, dataset, axis, tag, seed):
            print(f"[{index}/{len(matrix)}] SKIP existing {dataset}/{axis.name}/{tag}/seed_{seed}")
            continue
        print(f"[{index}/{len(matrix)}] {dataset}/{axis.name}/{tag}/seed_{seed}", flush=True)
        all_rows.extend(run_cell(args, dataset, axis, severity, seed, models, frozen, stamp))

    if all_rows:
        summary_path = args.output_root / f"all_metrics_{stamp}.csv"
        pd.DataFrame(all_rows).to_csv(summary_path, index=False)
        config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        config.update(
            {
                "n_cells": len(matrix),
                "python": sys.version,
                "platform": platform.platform(),
                "frozen_models": sorted(frozen),
            }
        )
        (args.output_root / f"sweep_config_{stamp}.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
