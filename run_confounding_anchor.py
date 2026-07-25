import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/home/jupyter/project/robustness_tests")
import run_robustness as rr
from robustness_tests.axes import AxisSpec
import numpy as np
from baseline_benchmark.data import _make_preprocessor

MODELS = ["t_learner", "x_learner", "dr_learner", "dragonnet", "causalpfn"]
EQUALIZE_ROWS = "40000"

AXIS_V1 = AxisSpec(
    name="confounding", severity_kind="conversion_lambda",
    default_grid=(0.0,), default_degrade_splits="train+val", extra_models=(),
    description="Approach 1 anchor, seed 42",
)


def make_confounding_fn_v1(strength, seed):
    def confounding_fn(X_slice):
        rng = np.random.default_rng(seed)
        preprocessor = _make_preprocessor(X_slice)
        matrix = np.asarray(preprocessor.fit_transform(X_slice), dtype=float)
        d = matrix.shape[1]
        k = min(5, d)
        chosen = rng.choice(d, size=k, replace=False)
        weights = rng.normal(size=k)
        signal = matrix[:, chosen] @ weights
        std = float(np.std(signal))
        signal = (signal - float(np.mean(signal))) / std if std > 1e-12 else np.zeros(len(signal))
        prob = 1.0 / (1.0 + np.exp(-strength * signal))
        return np.clip(prob, 1e-3, 1 - 1e-3)
    return confounding_fn


def patched_prepare_cell(args, dataset, axis, severity, seed):
    max_rows = None if args.max_rows == 0 else args.max_rows
    confounding_fn = make_confounding_fn_v1(float(severity), seed=seed * 1000 + int(float(severity) * 100))
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
rr.run_cell(args, "retailhero", AXIS_V1, 0.0, 42, MODELS, {}, stamp)
print("DONE v1 anchor seed42")
