import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "/home/jupyter/project/robustness_tests")

import run_robustness as rr
from robustness_tests.axes import AxisSpec

# ---- CONFIG (approach 2: risk-based confounding) ----
STRENGTHS = [0.0, 2.0, 4.0]
SEEDS = [0, 1]
DATASETS = ["retailhero"]
MODELS = ["t_learner", "x_learner", "dr_learner", "dragonnet", "causalpfn"]
EQUALIZE_ROWS = "40000"
# -------------------------------------------------------

from baseline_benchmark.data import _make_preprocessor, load_cleaned_dataset

CONFOUNDING_AXIS_V2 = AxisSpec(
    name="confounding_risk",
    severity_kind="conversion_lambda",
    default_grid=tuple(STRENGTHS),
    default_degrade_splits="train+val",
    extra_models=(),
    description=(
        "Approach 2: covariate-dependent P*(T=1|C) built from a fitted "
        "outcome-risk model (realistic, business-like targeting), contrasted "
        "with approach 1's arbitrary random-covariate confounder."
    ),
)

_RISK_MODEL_CACHE = {}


def _get_risk_scorer(cleaned_root, dataset, seed, max_rows):
    key = (dataset, seed, max_rows)
    if key in _RISK_MODEL_CACHE:
        return _RISK_MODEL_CACHE[key]
    X_ref, t_ref, y_ref, ids_ref, outcome_ref, group_safe_ref = load_cleaned_dataset(
        cleaned_root=cleaned_root, dataset=dataset, outcome=None, max_rows=max_rows, seed=seed
    )
    preprocessor = _make_preprocessor(X_ref)
    matrix_ref = np.asarray(preprocessor.fit_transform(X_ref), dtype=float)
    clf = HistGradientBoostingClassifier(random_state=seed, max_iter=100)
    clf.fit(matrix_ref, y_ref)
    _RISK_MODEL_CACHE[key] = (preprocessor, clf)
    return preprocessor, clf


def make_risk_confounding_fn(cleaned_root, dataset, seed, max_rows, strength):
    preprocessor, clf = _get_risk_scorer(cleaned_root, dataset, seed, max_rows)

    def confounding_fn(X_slice):
        matrix = np.asarray(preprocessor.transform(X_slice), dtype=float)
        risk = clf.predict_proba(matrix)[:, 1]
        std = float(np.std(risk))
        signal = (risk - float(np.mean(risk))) / std if std > 1e-12 else np.zeros(len(risk))
        logits = strength * signal
        prob = 1.0 / (1.0 + np.exp(-logits))
        return np.clip(prob, 1e-3, 1 - 1e-3)
    return confounding_fn


def patched_prepare_cell_v2(args, dataset, axis, severity, seed):
    max_rows = None if args.max_rows == 0 else args.max_rows
    confounding_fn = make_risk_confounding_fn(
        args.cleaned_root, dataset, seed, max_rows, float(severity)
    )
    return rr.prepare_resampled_data(
        cleaned_root=args.cleaned_root,
        dataset=dataset,
        outcome=args.outcome,
        max_rows=max_rows,
        axis="control_share",
        severity=0.5,
        seed=seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        degrade_splits=args.degrade_splits or axis.default_degrade_splits,
        conversion_risk=args.conversion_risk,
        conversion_risk_sign=args.conversion_risk_sign,
        run_diagnostics=not args.no_diagnostics,
        allow_confounding=True,
        equalize_rows=None if args.equalize_rows == 0 else args.equalize_rows,
        confounding_fn=confounding_fn,
    )


rr._prepare_cell = patched_prepare_cell_v2


def main():
    argv = [
        "--cleaned-root", "/home/jupyter/project/data_A_cleaned",
        "--datasets", ",".join(DATASETS),
        "--axes", "scale",
        "--seeds", "0",
        "--equalize-rows", EQUALIZE_ROWS,
        "--max-rows", "0",
        "--allow-confounding",
    ]
    args = rr.parse_args(argv)
    args.output_root = Path("/home/jupyter/project/robustness_tests/results_robustness")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dataset in DATASETS:
        for strength in STRENGTHS:
            for seed in SEEDS:
                print(f"=== {dataset} confounding_risk_lam{strength:.2f} seed {seed} ===", flush=True)
                try:
                    rr.run_cell(args, dataset, CONFOUNDING_AXIS_V2, strength, seed, MODELS, {}, stamp)
                except Exception as exc:
                    print(f"CELL FAILED {dataset}/confounding_risk/{strength}/seed_{seed}: {exc}", flush=True)
                    traceback.print_exc()


if __name__ == "__main__":
    main()
