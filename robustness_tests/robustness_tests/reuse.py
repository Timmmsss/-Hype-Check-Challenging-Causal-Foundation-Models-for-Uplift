"""Reuse completed baseline runs for the undegraded anchor of each axis.

The full-table CausalPFN evaluations from the earlier stage are expensive --
Criteo takes hours -- and the anchor severity of every axis is by definition the
*undegraded* data, which is exactly what those runs already evaluated. Rather
than recomputing them, this module ingests them.

Reuse is only sound when the earlier run drew the same split from the same rows.
``baseline_benchmark.data.prepare_data`` is deterministic given
``(dataset, outcome, max_rows, seed, val_fraction, test_fraction)``, so those
fields are compared against the earlier ``run_config.json`` and a mismatch is
refused loudly rather than silently relabelled. The caller additionally checks
that the evaluation row identifiers agree before mixing reused predictions with
freshly fitted ones.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

#: run_config.json fields that must agree for two runs to share a split and an
#: evaluation protocol. CausalPFN's context settings are included because they
#: change what the model saw, not just how fast it ran.
COMPATIBILITY_KEYS = (
    "outcome_resolved",
    "max_rows_resolved",
    "val_fraction",
    "test_fraction",
    "evaluation_split",
    "causalpfn_max_context_length",
    "causalpfn_max_query_length",
    "causalpfn_num_neighbours",
)


class IncompatibleRunError(RuntimeError):
    """An existing run does not match the protocol of the current sweep."""


def _load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.exists():
        raise IncompatibleRunError(f"{run_dir} has no run_config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def check_compatibility(config: dict[str, Any], requirements: dict[str, Any]) -> list[str]:
    """Return the keys on which an existing run disagrees with the sweep."""
    differences = []
    for key in COMPATIBILITY_KEYS:
        if key not in requirements:
            continue
        expected, actual = requirements[key], config.get(key)
        if isinstance(expected, float) or isinstance(actual, float):
            try:
                if abs(float(expected) - float(actual)) > 1e-9:
                    differences.append(f"{key}: run={actual!r} sweep={expected!r}")
                continue
            except (TypeError, ValueError):
                pass
        if expected != actual:
            differences.append(f"{key}: run={actual!r} sweep={expected!r}")
    return differences


def find_reusable_run(
    results_root: Path,
    dataset: str,
    seed: int,
    requirements: dict[str, Any],
    strict: bool = True,
) -> Optional[Path]:
    """Locate a completed run for this dataset and seed that matches the protocol.

    Both output layouts the earlier stage produced are searched: the plain
    ``<dataset>/seed_<seed>_<stamp>`` directories and the newer
    ``<dataset>/<split>_seed_<seed>_<stamp>`` ones.
    """
    results_root = Path(results_root)
    if not results_root.exists():
        return None

    candidates = [
        path
        for pattern in (f"**/{dataset}/*seed_{seed}_*", f"**/*seed_{seed}_*")
        for path in results_root.glob(pattern)
        if path.is_dir()
        and (path / "metrics.csv").exists()
        and (path / "predictions.parquet").exists()
    ]
    # Deduplicate while keeping the newest first: directory names end in a
    # timestamp, so a reverse sort puts the most recent run in front.
    unique = sorted({path.resolve() for path in candidates}, reverse=True)

    mismatches: list[str] = []
    for path in unique:
        try:
            config = _load_config(path)
        except IncompatibleRunError:
            continue
        if str(config.get("dataset", "")).lower() != dataset.lower():
            continue
        if int(config.get("seed", -1)) != int(seed):
            continue
        differences = check_compatibility(config, requirements)
        if not differences:
            return path
        mismatches.append(f"{path.name}: " + "; ".join(differences))

    if mismatches and strict:
        raise IncompatibleRunError(
            f"Found run(s) for {dataset} seed {seed} but none match the sweep protocol:\n  "
            + "\n  ".join(mismatches)
            + "\nFix the sweep flags to match, or drop --reuse-existing for this cell."
        )
    return None


def ingest_run(
    run_dir: Path,
    models: Sequence[str],
    extra_columns: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one completed run's metrics and predictions for the requested models.

    Returns ``(metrics, predictions)`` restricted to models the run actually
    contains, with the sweep's axis columns attached and a ``reused_from``
    provenance column so no reused row can be mistaken for a fresh one.
    """
    run_dir = Path(run_dir)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    predictions = pd.read_parquet(run_dir / "predictions.parquet")

    wanted = [name for name in models if name in set(metrics["model"].astype(str))]
    if not wanted:
        return pd.DataFrame(), pd.DataFrame()

    metrics = metrics.loc[metrics["model"].astype(str).isin(wanted)].copy()
    predictions = predictions.loc[predictions["model"].astype(str).isin(wanted)].copy()
    for column, value in extra_columns.items():
        metrics[column] = value
        if column in {"axis", "severity_tag"}:
            predictions[column] = value
    metrics["status"] = "ok"
    metrics["reused_from"] = str(run_dir)
    return metrics, predictions
