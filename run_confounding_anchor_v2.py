import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/home/jupyter/project/robustness_tests")
import run_robustness as rr
from robustness_tests.axes import AxisSpec
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from baseline_benchmark.data import _make_preprocessor, load_cleaned_dataset

MODELS = ["t_learner", "x_learner", "dr_learner", "dragonnet", "causalpfn"]
EQUALIZE_ROWS = "40000"

AXIS_V2 = AxisSpec(
    name="confounding_risk", severity_kind="conversion_lambda",
    default_grid=(0.0,), default_degrade_splits="train+val", extra_models=(),
    description="Approach 2 anchor, seed 42",
)


def make_risk_confounding_fn(cleaned_root, dataset, seed, max_rows, strength):
    X_ref, t_ref, y_ref, ids_ref, outcome_ref, group_safe_ref = load_cleaned_dataset(
        cleaned_root=cleaned_root, dataset=dataset, outcome=None, max_rows=max_rows, seed=seed
    )
    preprocessor = _make_preprocessor(X_ref)
    matrix_ref = np.asarray(preprocessor.fit_transform(X_ref), dtype=float)
    clf = HistGradientBoostingClassifier(random_state=seed, max_iter=100)
    clf.fit(matrix_ref, y_ref)

    def confounding_fn(X_slice):
        matrix = np.asarray(preprocessor.transform(X_slice), dtype=float)
        risk = clf.predict_proba(matrix)[:, 1]
        std = float(np.std(risk))
        signal = (risk - float(np.mean(risk))) / std if std > 1e-12 else np.zeros(len(risk))
        prob = 1.0 / (1.0 + np.exp(-strength * signal))
        return np.clip(prob, 1e-3, 1 - 1e-3)
    return confounding_fn


def patched_prepare_cell(args, dataset, axis, severity, seed):
    max_rows = None if args.max_rows == 0 else args.max_rows
    confounding_fn = make_risk_confounding_fn(args.cleaned_root, dataset, seed, max_rows, float(severity))
    return rr.prepare_resampled_data(
        cleaned_root=args.cleaned_root, dataset=dataset, outcome=args.outcome, max_rows=max_rows,
        axis="control_share", severity=0.5, seed=seed,
        val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        degrade_splits=args.degrade_splits or axis.default_degrade_splits,
        conversion_risk=args.conversion_risk, conversion_risk_sign=args.conversion_risk_sign,
        run_diagnostics=not args.no_diagnostics, allow_confounding=True,
        equalize_rows=None if args.equalize_rows == 0 else args.equalize_rows,
        confounding_fn=confounding_fn,
    )


rr._prepare_cell = patched_prepare_cell

argv = [
    "--cleaned-root", "/home/jupyter/project/data_A_cleaned",
    "--datasets", "retailhero", "--axes", "scale", "--seeds", "0",
    "--equalize-rows", EQUALIZE_ROWS, "--max-rows", "0", "--allow-confounding",
]
args = rr.parse_args(argv)
args.output_root = Path("/home/jupyter/project/robustness_tests/results_robustness")
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
rr.run_cell(args, "retailhero", AXIS_V2, 0.0, 42, MODELS, {}, stamp)
print("DONE v2 anchor seed42")
