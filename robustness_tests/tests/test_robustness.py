"""Tests for the robustness harness.

Everything here runs on synthetic data, so the suite needs no access to the
cleaned campaign datasets. Tests that require sklearn are skipped when it is
absent, which keeps the resampling and statistics testable anywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustness_tests.axes import AXES, get_axis, parse_grid, resolve_degrade_splits
from robustness_tests.bootstrap import bootstrap_cell, paired_difference, percentile_interval
from robustness_tests.diagnostics import (
    benjamini_hochberg,
    check_event_counts,
    check_outcome_dependent_covariate,
)
from robustness_tests.pehe import abs_ate_error, average_ranks, pehe, ranking_agreement, spearman
from robustness_tests.resampling import (
    control_share_rule,
    conversion_accept_probability,
    covariate_selection_sample,
    rejection_sample,
    uniform_subsample_rule,
)

def synthetic_rct(seed=7, n=4000, d=6):
    """Same shape as ``baseline_benchmark/tests/test_smoke.py::synthetic_rct``."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    t = rng.binomial(1, 0.5, size=n).astype(np.int8)
    tau = 0.18 * np.tanh(X[:, 0])
    p0 = 1 / (1 + np.exp(-(-2.0 + 0.4 * X[:, 1])))
    p = np.clip(p0 + t * tau, 0.001, 0.999)
    y = rng.binomial(1, p).astype(np.int8)
    return X, t, y, tau


# --------------------------------------------------------------------------
# Axes
# --------------------------------------------------------------------------
def test_every_axis_grid_starts_undegraded():
    for name, axis in AXES.items():
        assert axis.is_anchor(axis.default_grid[0]), f"{name} grid must open with an anchor"


def test_severity_order_ranks_more_degraded_higher():
    scale = get_axis("scale")
    assert scale.severity_order(None) < scale.severity_order(50_000)
    assert scale.severity_order(50_000) < scale.severity_order(5_000)
    control = get_axis("control_share")
    assert control.severity_order(0.5) < control.severity_order(0.99)


def test_degrade_split_defaults_match_the_protocol():
    # Qini needs a stable evaluation population on the scale and control-share
    # axes; low conversion is a property of the deployed population.
    assert resolve_degrade_splits(get_axis("scale"), None) == ("train", "validation")
    assert resolve_degrade_splits(get_axis("control_share"), None) == ("train", "validation")
    assert resolve_degrade_splits(get_axis("conversion"), None) == (
        "train",
        "validation",
        "test",
    )


def test_parse_grid_round_trips_and_rejects_bad_values():
    control = get_axis("control_share")
    assert parse_grid(control, "0.5,0.9") == [0.5, 0.9]
    with pytest.raises(ValueError):
        parse_grid(control, "1.5")
    assert parse_grid(get_axis("scale"), "full,1000") == [None, 1000]


def test_severity_tags_are_directory_safe_and_unique():
    for axis in AXES.values():
        tags = [axis.tag(value) for value in axis.default_grid]
        assert len(set(tags)) == len(tags)
        for tag in tags:
            assert "." not in tag and "/" not in tag


# --------------------------------------------------------------------------
# Resampling: the identification-critical properties
# --------------------------------------------------------------------------
def test_control_share_hits_the_target_rate():
    _, t, y, _ = synthetic_rct(n=20_000)
    rng = np.random.default_rng(0)
    for target in (0.7, 0.9, 0.95):
        keep, record = control_share_rule(rng, t, target, y=y)
        assert abs(record.treatment_rate_after - target) < 0.02
        assert record.n_after < record.n_before
        assert record.preserves_randomization


def test_control_share_preserves_the_covariate_distribution():
    """Constant P*(T|C) thins uniformly inside an arm, so P(C) must survive."""
    X, t, y, _ = synthetic_rct(n=40_000)
    keep, _ = control_share_rule(np.random.default_rng(1), t, 0.9, y=y)
    for column in range(X.shape[1]):
        before, after = X[:, column], X[keep, column]
        # Difference in means, in standard errors of the resampled mean.
        z = abs(after.mean() - before.mean()) / (after.std() / np.sqrt(len(after)))
        assert z < 4.0, f"covariate {column} shifted (z={z:.2f})"


def test_control_share_preserves_outcome_given_treatment_and_covariates():
    _, t, y, _ = synthetic_rct(n=40_000)
    keep, _ = control_share_rule(np.random.default_rng(2), t, 0.9, y=y)
    for arm in (0, 1):
        before = y[t == arm].mean()
        after = y[keep][t[keep] == arm].mean()
        assert abs(before - after) < 0.01


def test_conversion_rule_never_depends_on_the_outcome():
    """Keith et al.'s Proposition 3.1 constraint, asserted directly.

    Permuting Y must leave the accepted row set bit-identical: the acceptance
    probability is a function of covariates alone.
    """
    X, t, y, _ = synthetic_rct(n=8000)
    probability = conversion_accept_probability(X[:, 1], lam=3.0)
    keep_a, _ = covariate_selection_sample(np.random.default_rng(5), probability, t, y=y)
    shuffled = np.random.default_rng(99).permutation(y)
    keep_b, _ = covariate_selection_sample(np.random.default_rng(5), probability, t, y=shuffled)
    assert np.array_equal(keep_a, keep_b)


def test_conversion_rule_lowers_the_base_rate_monotonically():
    X, t, y, _ = synthetic_rct(n=40_000)
    # Baseline risk rises with X[:, 1] in the generator above.
    risk = X[:, 1]
    rates = []
    for lam in (0.0, 1.0, 2.0, 4.0):
        probability = conversion_accept_probability(risk, lam=lam)
        keep, record = covariate_selection_sample(np.random.default_rng(3), probability, t, y=y)
        rates.append(record.outcome_rate_after)
    assert rates == sorted(rates, reverse=True), rates
    assert rates[-1] < 0.75 * rates[0]


def test_conversion_rule_leaves_the_treatment_rate_alone():
    X, t, y, _ = synthetic_rct(n=40_000)
    probability = conversion_accept_probability(X[:, 1], lam=4.0)
    _, record = covariate_selection_sample(np.random.default_rng(4), probability, t, y=y)
    assert abs(record.treatment_rate_after - record.treatment_rate_before) < 0.02
    assert record.preserves_randomization


def test_conversion_accept_probability_is_identity_at_zero():
    probability = conversion_accept_probability(np.arange(10.0), lam=0.0)
    assert np.allclose(probability, 1.0)


def test_conversion_accept_probability_ties_share_a_value():
    probability = conversion_accept_probability(np.array([1.0, 1.0, 2.0, 3.0]), lam=2.0)
    assert probability[0] == pytest.approx(probability[1])


def test_rejection_sample_flags_confounding_when_p_star_varies():
    X, t, y, _ = synthetic_rct(n=10_000)
    star = np.clip(0.5 + 0.3 * np.tanh(X[:, 0]), 0.05, 0.95)
    _, record = rejection_sample(np.random.default_rng(6), t, star, y=y)
    assert not record.preserves_randomization
    assert not record.conditions_on_outcome


def test_rejection_sample_rejects_a_propensity_outside_the_unit_interval():
    _, t, _, _ = synthetic_rct(n=500)
    with pytest.raises(ValueError, match="positivity"):
        rejection_sample(np.random.default_rng(0), t, propensity_star=1.0)


def test_rejection_sample_rejects_too_small_an_envelope():
    _, t, _, _ = synthetic_rct(n=500)
    with pytest.raises(ValueError, match="envelope_M"):
        rejection_sample(np.random.default_rng(0), t, propensity_star=0.9, envelope_M=0.5)


def test_uniform_subsample_keeps_everything_at_the_anchor():
    _, t, y, _ = synthetic_rct(n=1000)
    keep, record = uniform_subsample_rule(np.random.default_rng(0), t, None, y=y)
    assert len(keep) == len(t)
    assert record.acceptance_rate == 1.0
    keep, record = uniform_subsample_rule(np.random.default_rng(0), t, 250, y=y)
    assert len(keep) == 250


def test_resampling_is_reproducible_for_a_fixed_seed():
    _, t, y, _ = synthetic_rct(n=5000)
    first, _ = control_share_rule(np.random.default_rng(11), t, 0.9, y=y)
    second, _ = control_share_rule(np.random.default_rng(11), t, 0.9, y=y)
    assert np.array_equal(first, second)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def test_precondition_passes_when_a_covariate_predicts_the_outcome():
    X, _, y, _ = synthetic_rct(n=5000)
    result = check_outcome_dependent_covariate(X, y)
    assert result["precondition_met"]
    assert result["n_dependent"] >= 1


def test_precondition_fails_on_independent_covariates():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5))
    y = rng.binomial(1, 0.2, size=4000)
    with pytest.raises(ValueError, match="precondition"):
        check_outcome_dependent_covariate(X, y, strict=True)
    reported = check_outcome_dependent_covariate(X, y, strict=False)
    assert not reported["precondition_met"]


def test_benjamini_hochberg_controls_the_obvious_cases():
    assert benjamini_hochberg([0.001, 0.002, 0.9]).tolist() == [True, True, False]
    assert not benjamini_hochberg([0.6, 0.7, 0.8]).any()


def test_event_counts_flag_a_degenerate_cell():
    t = np.array([0] * 50 + [1] * 50)
    y = np.zeros(100)
    y[:3] = 1  # three events, all in the control arm
    check = check_event_counts(t, y, "test")
    assert check["degenerate"]
    assert check["both_arms_present"]
    assert check["arm_1"]["events"] == 0


def test_event_counts_accept_a_healthy_cell():
    _, t, y, _ = synthetic_rct(n=20_000)
    check = check_event_counts(t, y, "train")
    assert not check["degenerate"]


# --------------------------------------------------------------------------
# PEHE and ranking agreement
# --------------------------------------------------------------------------
def test_pehe_is_zero_for_a_perfect_estimator():
    tau = np.array([0.1, -0.2, 0.3])
    assert pehe(tau, tau) == pytest.approx(0.0)
    assert abs_ate_error(tau, tau) == pytest.approx(0.0)


def test_pehe_matches_a_hand_computation():
    assert pehe([0.0, 0.0], [1.0, -1.0]) == pytest.approx(1.0)
    assert abs_ate_error([0.0, 0.0], [1.0, -1.0]) == pytest.approx(0.0)


def test_average_ranks_share_ties():
    assert average_ranks([3.0, 1.0, 1.0]).tolist() == [3.0, 1.5, 1.5]


def test_spearman_recovers_perfect_agreement():
    rho, p = spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert rho == pytest.approx(1.0)
    assert p < 0.05


def test_spearman_recovers_perfect_disagreement():
    rho, _ = spearman([1, 2, 3, 4], [4, 3, 2, 1])
    assert rho == pytest.approx(-1.0)


def test_ranking_agreement_is_positive_when_criteria_agree():
    # Lowest PEHE and highest Qini identify the same best method.
    result = ranking_agreement(
        {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4},
        {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1},
    )
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["pehe_order"][0] == "a" and result["qini_order"][0] == "a"


def test_ranking_agreement_is_negative_when_criteria_conflict():
    result = ranking_agreement(
        {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4},
        {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4},
    )
    assert result["spearman_rho"] == pytest.approx(-1.0)


def test_ranking_agreement_needs_three_methods():
    result = ranking_agreement({"a": 0.1, "b": 0.2}, {"a": 0.2, "b": 0.1})
    assert np.isnan(result["spearman_rho"])
    assert result["n_methods"] == 2


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
def _predictions_frame(seed=0, n=3000, n_models=3, noise_levels=None):
    X, t, y, tau = synthetic_rct(seed=seed, n=n)
    rng = np.random.default_rng(seed + 100)
    levels = noise_levels or [(index + 1) * 0.5 for index in range(n_models)]
    frames = []
    for index, level in enumerate(levels):
        noise = level * rng.normal(size=n)
        frames.append(
            pd.DataFrame(
                {
                    "epk_id": np.arange(n),
                    "dataset": "synthetic",
                    "axis": "scale",
                    "severity_tag": "scale_full",
                    "model": f"model_{index}",
                    "seed": seed,
                    "T": t,
                    "Y": y,
                    "cate_pred": tau + noise,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_bootstrap_interval_brackets_the_point_estimate():
    result = bootstrap_cell(_predictions_frame(), n_bootstrap=200, seed=1)
    for model in result["models"]:
        point = result["point"]["qini_auc_normalized"][model]
        interval = percentile_interval(result["draws"]["qini_auc_normalized"][model])
        assert interval["ci_low"] <= point <= interval["ci_high"], model


def test_bootstrap_point_estimate_matches_the_library_metric():
    from baseline_benchmark.metrics import qini_auc_normalized

    frame = _predictions_frame()
    result = bootstrap_cell(frame, n_bootstrap=10, seed=1)
    subset = frame.loc[frame["model"] == "model_0"].sort_values("epk_id")
    expected = qini_auc_normalized(
        subset["Y"].to_numpy(float),
        subset["cate_pred"].to_numpy(float),
        subset["T"].to_numpy(float),
    )
    assert result["point"]["qini_auc_normalized"]["model_0"] == pytest.approx(expected)


def test_bootstrap_is_reproducible():
    first = bootstrap_cell(_predictions_frame(), n_bootstrap=100, seed=7)
    second = bootstrap_cell(_predictions_frame(), n_bootstrap=100, seed=7)
    assert np.allclose(
        first["draws"]["qini_auc_normalized"]["model_0"],
        second["draws"]["qini_auc_normalized"]["model_0"],
        equal_nan=True,
    )


def test_paired_difference_detects_a_better_model():
    # model_0 sees the true effect; model_1 is essentially noise.
    frame = _predictions_frame(n=8000, noise_levels=[0.0, 5.0])
    result = bootstrap_cell(frame, n_bootstrap=300, seed=2)
    difference = paired_difference(
        result["draws"]["qini_auc_normalized"], "model_0", "model_1"
    )
    assert difference["mean_difference"] > 0.0
    assert difference["excludes_zero"]


def test_paired_difference_is_tighter_than_an_unpaired_comparison():
    """Pairing on identical resampled rows is what buys the power for H3.

    Both models share most of their bootstrap variation, so the paired interval
    on the difference must be narrower than combining the two marginal
    intervals independently.
    """
    result = bootstrap_cell(_predictions_frame(n=6000), n_bootstrap=300, seed=3)
    draws = result["draws"]["qini_auc_normalized"]
    paired_width = np.nanstd(draws["model_0"] - draws["model_2"])
    unpaired_width = np.sqrt(
        np.nanvar(draws["model_0"]) + np.nanvar(draws["model_2"])
    )
    assert paired_width < unpaired_width


def test_bootstrap_rejects_mismatched_evaluation_rows():
    frame = _predictions_frame(n=1000)
    frame = frame.drop(frame.loc[frame["model"] == "model_1"].index[:5])
    with pytest.raises(ValueError, match="different row set"):
        bootstrap_cell(frame, n_bootstrap=5, seed=0)


# --------------------------------------------------------------------------
# Estimators and the full pipeline (these need sklearn)
# --------------------------------------------------------------------------
def _skip_without_sklearn():
    pytest.importorskip("sklearn")


def _ratio_dgp(seed=3, n=6000, d=5):
    """Binary outcomes with a genuinely heterogeneous ratio effect."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    t = rng.binomial(1, 0.5, size=n).astype(np.int8)
    p0 = 1 / (1 + np.exp(-(-2.2 + 0.8 * X[:, 0])))
    ratio = np.exp(0.6 + 0.8 * np.tanh(X[:, 1]))
    p1 = np.clip(p0 * ratio, 1e-3, 1 - 1e-3)
    y = rng.binomial(1, np.where(t == 1, p1, p0)).astype(np.int8)
    return X, t, y, p1 - p0, p1 / p0


def test_q_learner_recovers_the_ratio_ordering():
    _skip_without_sklearn()
    from robustness_tests.qlearner import QLearnerRatio

    X, t, y, _, true_ratio = _ratio_dgp(n=20_000)
    model = QLearnerRatio(seed=0, max_iter=80).fit(X, t, y)
    predicted = model.predict_cate(X)
    correlation = np.corrcoef(average_ranks(predicted), average_ranks(true_ratio))[0, 1]
    assert correlation > 0.5, correlation


def test_q_learner_difference_scale_tracks_the_true_difference():
    _skip_without_sklearn()
    from robustness_tests.qlearner import QLearner

    X, t, y, true_difference, _ = _ratio_dgp(n=20_000)
    model = QLearner(seed=0, max_iter=80).fit(X, t, y)
    predicted = model.predict_cate(X)
    assert np.isfinite(predicted).all()
    correlation = np.corrcoef(average_ranks(predicted), average_ranks(true_difference))[0, 1]
    assert correlation > 0.4, correlation


def test_q_learner_refuses_a_sample_with_too_few_converters():
    _skip_without_sklearn()
    from robustness_tests.qlearner import QLearner

    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    t = rng.binomial(1, 0.5, size=500).astype(np.int8)
    y = np.zeros(500, dtype=np.int8)
    y[:3] = 1
    with pytest.raises(ValueError, match="converters"):
        QLearner(seed=0, max_iter=10).fit(X, t, y)


def test_q_learner_refuses_converters_in_only_one_arm():
    _skip_without_sklearn()
    from robustness_tests.qlearner import QLearner

    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    t = rng.binomial(1, 0.5, size=500).astype(np.int8)
    y = np.zeros(500, dtype=np.int8)
    y[t == 1] = (rng.random(int((t == 1).sum())) < 0.5).astype(np.int8)
    with pytest.raises(ValueError, match="both treatment arms"):
        QLearner(seed=0, max_iter=10, min_converters=5).fit(X, t, y)


def test_gp_cate_recovers_a_smooth_effect_with_a_small_control_arm():
    _skip_without_sklearn()
    from robustness_tests.gpcate import GPCATE

    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    # Few placebos: 30 controls against 370 treated.
    t = np.ones(n, dtype=np.int8)
    t[rng.choice(n, size=30, replace=False)] = 0
    p0 = 1 / (1 + np.exp(-(-0.5 + 0.5 * X[:, 0])))
    tau = 0.25 * np.tanh(X[:, 0])
    p1 = np.clip(p0 + tau, 1e-3, 1 - 1e-3)
    y = rng.binomial(1, np.where(t == 1, p1, p0)).astype(np.int8)

    model = GPCATE(seed=0, max_control=30, max_treated=120, n_components=3).fit(X, t, y)
    predicted = model.predict_cate(X)
    assert predicted.shape == (n,)
    assert np.isfinite(predicted).all()
    correlation = np.corrcoef(average_ranks(predicted), average_ranks(tau))[0, 1]
    assert correlation > 0.3, correlation


def test_gp_cate_intervals_are_ordered_and_widen_with_the_level():
    _skip_without_sklearn()
    from robustness_tests.gpcate import GPCATE

    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    t = rng.binomial(1, 0.5, size=200).astype(np.int8)
    y = rng.binomial(1, 0.3, size=200).astype(np.int8)
    model = GPCATE(seed=0, max_control=80, max_treated=80, n_components=3).fit(X, t, y)
    low, high = model.predict_cate_interval(X, level=0.95)
    narrow_low, narrow_high = model.predict_cate_interval(X, level=0.5)
    assert (low <= high).all()
    assert ((high - low) >= (narrow_high - narrow_low)).all()


def test_registry_delegates_without_touching_the_baseline_model_list():
    _skip_without_sklearn()
    from baseline_benchmark.models import available_models
    from robustness_tests.registry import available_robustness_models, make_robustness_model

    # The registry must delegate without leaking its own estimators into the
    # baseline list. The five core models must all still be offered there; the
    # baseline set may legitimately grow (e.g. CausalPFN fine-tuning variants),
    # so this checks membership rather than an exact list.
    baseline = available_models()
    for core in ("t_learner", "x_learner", "dr_learner", "dragonnet", "causalpfn"):
        assert core in baseline, core
    assert "q_learner" not in baseline and "gp_cate" not in baseline
    assert "q_learner" in available_robustness_models()
    assert make_robustness_model("t_learner", seed=1, max_iter=5).name == "t_learner"
    assert make_robustness_model("q_learner", seed=1, max_iter=5).name == "q_learner"
    with pytest.raises(ValueError, match="Unknown model"):
        make_robustness_model("not_a_model")


def test_supports_pehe_excludes_the_ratio_scale():
    from robustness_tests.registry import supports_pehe

    assert supports_pehe("q_learner")
    assert not supports_pehe("q_learner_ratio")


# --------------------------------------------------------------------------
# Semi-synthetic generator
# --------------------------------------------------------------------------
def test_semisynthetic_hits_the_requested_base_rate_and_ate():
    _skip_without_sklearn()
    from robustness_tests.semisynth import SemiSyntheticSpec, load_covariates, make_binary_semisynthetic

    spec = SemiSyntheticSpec(
        covariate_source="gaussian:20000x8", base_rate=0.04, ate=0.02, seed=1
    )
    rng = np.random.default_rng(1)
    X, _ = load_covariates(spec.covariate_source, rng)
    t, y, tau, info = make_binary_semisynthetic(X, spec, rng)
    assert info["realized_base_rate_control"] == pytest.approx(0.04, abs=0.005)
    assert info["realized_ate_true"] == pytest.approx(0.02, abs=0.005)
    assert set(np.unique(y)).issubset({0, 1})
    assert tau.shape == (len(X),)
    assert tau.std() > 0.0, "the generator must produce heterogeneous effects"


def test_semisynthetic_ground_truth_is_exact_after_clipping():
    _skip_without_sklearn()
    from robustness_tests.semisynth import SemiSyntheticSpec, load_covariates, make_binary_semisynthetic

    # A large ATE forces clipping; tau must still equal p1 - p0 exactly.
    spec = SemiSyntheticSpec(covariate_source="gaussian:5000x6", base_rate=0.5, ate=0.9, seed=2)
    rng = np.random.default_rng(2)
    X, _ = load_covariates(spec.covariate_source, rng)
    _, _, tau, _ = make_binary_semisynthetic(X, spec, rng)
    assert (tau >= -1.0).all() and (tau <= 1.0).all()
    assert np.isfinite(tau).all()


def test_semisynthetic_pipeline_carries_ground_truth_into_every_split():
    _skip_without_sklearn()
    from robustness_tests.semisynth import SemiSyntheticSpec, prepare_semisynthetic_data

    spec = SemiSyntheticSpec(covariate_source="gaussian:6000x6", base_rate=0.1, ate=0.03, seed=3)
    resampled = prepare_semisynthetic_data(spec, axis="scale", severity=1500, seed=0)
    data = resampled.prepared
    assert len(data.y_train) == 1500
    for split, expected in (
        ("train", len(data.y_train)),
        ("validation", len(data.y_val)),
        ("test", len(data.y_test)),
    ):
        tau = resampled.true_cate(split)
        assert tau is not None and len(tau) == expected


def test_scale_axis_leaves_the_evaluation_split_identical_across_severities():
    """The whole point of degrading train+val only: comparable Qini."""
    _skip_without_sklearn()
    from robustness_tests.semisynth import SemiSyntheticSpec, prepare_semisynthetic_data

    spec = SemiSyntheticSpec(covariate_source="gaussian:6000x6", base_rate=0.1, ate=0.03, seed=4)
    small = prepare_semisynthetic_data(spec, axis="scale", severity=1000, seed=0)
    large = prepare_semisynthetic_data(spec, axis="scale", severity=2500, seed=0)
    assert np.array_equal(small.prepared.id_test, large.prepared.id_test)
    assert np.array_equal(small.prepared.y_test, large.prepared.y_test)
    assert len(small.prepared.y_train) == 1000
    assert len(large.prepared.y_train) == 2500


def test_control_share_axis_keeps_the_evaluation_split_balanced():
    _skip_without_sklearn()
    from robustness_tests.semisynth import SemiSyntheticSpec, prepare_semisynthetic_data

    spec = SemiSyntheticSpec(covariate_source="gaussian:8000x6", base_rate=0.15, ate=0.04, seed=5)
    resampled = prepare_semisynthetic_data(spec, axis="control_share", severity=0.95, seed=0)
    assert resampled.prepared.t_train.mean() == pytest.approx(0.95, abs=0.03)
    # Test is untouched by default on this axis.
    assert resampled.prepared.t_test.mean() == pytest.approx(0.5, abs=0.05)


def test_conversion_axis_degrades_every_split_by_default():
    _skip_without_sklearn()
    from robustness_tests.semisynth import SemiSyntheticSpec, prepare_semisynthetic_data

    spec = SemiSyntheticSpec(covariate_source="gaussian:12000x6", base_rate=0.2, ate=0.03, seed=6)
    anchor = prepare_semisynthetic_data(spec, axis="conversion", severity=0.0, seed=0)
    degraded = prepare_semisynthetic_data(spec, axis="conversion", severity=4.0, seed=0)
    assert degraded.prepared.y_train.mean() < anchor.prepared.y_train.mean()
    assert degraded.prepared.y_test.mean() < anchor.prepared.y_test.mean()


def test_conversion_axis_moves_train_and_eval_by_similar_amounts():
    """Out-of-fold scoring of the training rows removes a sharpness artifact.

    Scored in sample, the risk model separates training rows far better than
    held-out rows, so the same knob would thin training conversions much harder
    than evaluation conversions -- a property of the model, not of the regime.
    """
    _skip_without_sklearn()
    from robustness_tests.semisynth import SemiSyntheticSpec, prepare_semisynthetic_data

    spec = SemiSyntheticSpec(covariate_source="gaussian:20000x8", base_rate=0.15, ate=0.03, seed=8)
    anchor = prepare_semisynthetic_data(spec, axis="conversion", severity=0.0, seed=0)
    degraded = prepare_semisynthetic_data(spec, axis="conversion", severity=5.0, seed=0)

    train_ratio = degraded.prepared.y_train.mean() / anchor.prepared.y_train.mean()
    eval_ratio = degraded.prepared.y_test.mean() / anchor.prepared.y_test.mean()
    assert train_ratio < 0.9 and eval_ratio < 0.9, (train_ratio, eval_ratio)
    assert abs(train_ratio - eval_ratio) < 0.2, (train_ratio, eval_ratio)


def test_equalize_rows_holds_sample_size_fixed_across_severities():
    """Otherwise the conversion axis is confounded with the scale axis."""
    _skip_without_sklearn()
    from robustness_tests.semisynth import SemiSyntheticSpec, prepare_semisynthetic_data

    spec = SemiSyntheticSpec(covariate_source="gaussian:20000x8", base_rate=0.2, ate=0.03, seed=9)
    sizes = []
    for severity in (2.0, 6.0):
        resampled = prepare_semisynthetic_data(
            spec, axis="conversion", severity=severity, seed=0, equalize_rows=1500
        )
        sizes.append(len(resampled.prepared.y_train))
        # Validation and test are trimmed in the same 60/20/20 proportion.
        assert len(resampled.prepared.y_test) == 500
        assert resampled.diagnostics["equalization_reached"]
    assert sizes == [1500, 1500]


def test_equalization_reports_when_the_target_cannot_be_met():
    """A cap cannot invent rows; the shortfall has to be visible, not silent."""
    _skip_without_sklearn()
    from robustness_tests.semisynth import SemiSyntheticSpec, prepare_semisynthetic_data

    spec = SemiSyntheticSpec(covariate_source="gaussian:20000x8", base_rate=0.2, ate=0.03, seed=9)
    resampled = prepare_semisynthetic_data(
        spec, axis="conversion", severity=9.0, seed=0, equalize_rows=11_000
    )
    assert not resampled.diagnostics["equalization_reached"]


def test_equalize_rows_does_not_undo_the_regime():
    _skip_without_sklearn()
    from robustness_tests.semisynth import SemiSyntheticSpec, prepare_semisynthetic_data

    spec = SemiSyntheticSpec(covariate_source="gaussian:20000x8", base_rate=0.2, ate=0.03, seed=10)
    mild = prepare_semisynthetic_data(
        spec, axis="conversion", severity=2.0, seed=0, equalize_rows=3000
    )
    harsh = prepare_semisynthetic_data(
        spec, axis="conversion", severity=9.0, seed=0, equalize_rows=3000
    )
    assert harsh.prepared.y_train.mean() < mild.prepared.y_train.mean()


def test_confounding_requires_an_explicit_opt_in():
    _skip_without_sklearn()
    from robustness_tests.semisynth import SemiSyntheticSpec, prepare_semisynthetic_data

    spec = SemiSyntheticSpec(covariate_source="gaussian:3000x5", seed=7)
    with pytest.raises(ValueError, match="allow_confounding"):
        prepare_semisynthetic_data(
            spec, axis="control_share", severity=0.7, seed=0,
            confounding_fn=lambda frame: np.full(len(frame), 0.5),
        )


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def test_dry_run_prints_the_expected_matrix(capsys):
    _skip_without_sklearn()
    from run_robustness import main

    exit_code = main(
        [
            "--datasets", "semisynth_test",
            "--axes", "control_share",
            "--severities", "0.5,0.9",
            "--seeds", "0,1",
            "--models", "t_learner,x_learner",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "4 cells" in output
    assert "8 model fits" in output


def test_model_variants_default_when_no_flag_given():
    from run_robustness import parse_args, _model_variants, _fit_count

    args = parse_args(["--datasets", "semisynth_x"])
    for model in ("t_learner", "causalpfn", "gp_cate", "q_learner"):
        variants = _model_variants(args, model)
        assert len(variants) == 1
        assert variants[0].model_id == model
        assert variants[0].tag == "default"
    # One fit per model with no variants.
    assert _fit_count(args, ["t_learner", "gp_cate"]) == 2


def test_model_variants_expand_and_keep_the_default_plain():
    from run_robustness import parse_args, _model_variants, _fit_count

    args = parse_args(
        [
            "--datasets", "semisynth_x",
            "--gp-component-variants", "3,5",  # default is 20
            "--causalpfn-context-variants", "1024,4096",  # default is 4096
            "--q-min-converters-variants", "50,100",  # default is 50
        ]
    )
    gp = {variant.model_id: variant for variant in _model_variants(args, "gp_cate")}
    # 3, 5 and the default 20 -> three fits; the default keeps the plain name.
    assert set(gp) == {"gp_cate", "gp_cate#ncomp3", "gp_cate#ncomp5"}
    assert gp["gp_cate"].overrides == {} and gp["gp_cate"].hparam_value == 20
    assert gp["gp_cate#ncomp3"].overrides == {"n_components": 3}

    context = {variant.model_id for variant in _model_variants(args, "causalpfn")}
    # 4096 equals the default, so it is the plain series; only 1024 is suffixed.
    assert context == {"causalpfn", "causalpfn#ctx1024"}

    # q_learner and q_learner_ratio share the one flag.
    for model in ("q_learner", "q_learner_ratio"):
        ids = {variant.model_id for variant in _model_variants(args, model)}
        assert ids == {model, f"{model}#minconv100"}

    assert _fit_count(args, ["t_learner", "gp_cate"]) == 1 + 3


def test_variant_sweep_writes_hyperparameter_columns(tmp_path):
    _skip_without_sklearn()
    from run_robustness import main

    exit_code = main(
        [
            "--datasets", "semisynth_smoke",
            "--axes", "control_share",
            "--severities", "0.5,0.9",
            "--seeds", "0,1",
            "--models", "t_learner,gp_cate",
            "--gp-component-variants", "3,5",
            # Small GP caps keep the exact O(n^3) fit fast in tests.
            "--gp-max-control", "120",
            "--gp-max-treated", "80",
            "--semisynth-covariates", "gaussian:6000x6",
            "--semisynth-base-rate", "0.2",
            "--semisynth-ate", "0.05",
            "--tree-max-iter", "20",
            "--output-root", str(tmp_path),
            "--no-diagnostics",
        ]
    )
    assert exit_code == 0

    metrics = pd.concat(
        [pd.read_csv(path) for path in tmp_path.glob("**/metrics.csv")], ignore_index=True
    )
    for column in ("base_model", "hparam_tag", "hparam_name", "hparam_value", "hparams_used"):
        assert column in metrics.columns, column
    # gp_cate default (20) stays plain; 3 and 5 get suffixed series ids.
    gp_ids = set(metrics.loc[metrics["base_model"] == "gp_cate", "model"])
    assert gp_ids == {"gp_cate", "gp_cate#ncomp3", "gp_cate#ncomp5"}
    # t_learner is not swept, so it keeps its single default series.
    assert set(metrics.loc[metrics["base_model"] == "t_learner", "hparam_tag"]) == {"default"}
    # The realized hyperparameters are logged for a fitted gp_cate row.
    fitted = metrics.loc[(metrics["model"] == "gp_cate#ncomp3") & (metrics["status"] == "ok")]
    assert not fitted.empty
    assert json.loads(fitted.iloc[0]["hparams_used"]) ["n_components"] == 3

    # Every variant lands, paired, in one predictions file per cell.
    predictions = pd.concat(
        [pd.read_parquet(path) for path in tmp_path.glob("**/predictions.parquet")],
        ignore_index=True,
    )
    assert "hparam_tag" in predictions.columns
    assert {"gp_cate", "gp_cate#ncomp3", "gp_cate#ncomp5"}.issubset(set(predictions["model"]))


def test_analysis_reports_hyperparameter_influence(tmp_path):
    _skip_without_sklearn()
    from analyze_robustness import main as analyze_main
    from run_robustness import main as sweep_main

    sweep_main(
        [
            "--datasets", "semisynth_smoke",
            "--axes", "control_share",
            "--severities", "0.5,0.9,0.98",
            "--seeds", "0,1",
            "--models", "t_learner,x_learner,gp_cate",
            "--gp-component-variants", "3,5",
            "--gp-max-control", "120",
            "--gp-max-treated", "80",
            "--semisynth-covariates", "gaussian:6000x6",
            "--semisynth-base-rate", "0.2",
            "--semisynth-ate", "0.05",
            "--tree-max-iter", "20",
            "--output-root", str(tmp_path),
            "--no-diagnostics",
        ]
    )
    exit_code = analyze_main(
        [
            "--results-root", str(tmp_path),
            "--n-bootstrap", "50",
            "--reference-model", "gp_cate",
            "--no-figures",
        ]
    )
    assert exit_code == 0

    analysis = tmp_path / "analysis"
    per_variant = analysis / "hparam_breakpoints.csv"
    assert per_variant.exists(), "a swept model must produce a per-variant breakpoint table"
    frame = pd.read_csv(per_variant)
    # Every gp_cate variant is represented, and the default one is flagged.
    assert set(frame["hparam_tag"]) >= {"default", "ncomp3", "ncomp5"}
    assert frame.loc[frame["hparam_tag"] == "default", "is_default"].all()
    text = (analysis / "regime_map.md").read_text(encoding="utf-8")
    assert "Hyperparameter influence on the breakpoint" in text


def test_end_to_end_sweep_writes_a_usable_schema(tmp_path):
    _skip_without_sklearn()
    from run_robustness import main

    exit_code = main(
        [
            "--datasets", "semisynth_smoke",
            "--axes", "control_share",
            "--severities", "0.5,0.9",
            "--seeds", "0,1",
            "--models", "t_learner,x_learner",
            "--semisynth-covariates", "gaussian:6000x6",
            "--semisynth-base-rate", "0.2",
            "--semisynth-ate", "0.05",
            "--tree-max-iter", "20",
            "--output-root", str(tmp_path),
            "--no-diagnostics",
        ]
    )
    assert exit_code == 0

    metric_files = sorted(tmp_path.glob("**/metrics.csv"))
    assert len(metric_files) == 4
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    for column in (
        "axis", "severity", "severity_tag", "seed", "model", "status",
        "qini_auc_normalized", "pehe", "abs_ate_error", "train_treatment_rate",
    ):
        assert column in metrics.columns, column
    assert (metrics["status"] == "ok").all()
    assert set(metrics["axis"]) == {"control_share"}
    assert len(sorted(tmp_path.glob("**/predictions.parquet"))) == 4
    assert len(sorted(tmp_path.glob("**/resample_manifest.json"))) == 4


# --------------------------------------------------------------------------
# Reuse of completed baseline runs
# --------------------------------------------------------------------------
def _fake_completed_run(directory: Path, dataset="retailhero", seed=0, models=("causalpfn",)):
    """A minimal stand-in for a directory written by run_baselines.py."""
    directory.mkdir(parents=True, exist_ok=True)
    n = 400
    rng = np.random.default_rng(seed)
    frames, rows = [], []
    for model in models:
        frames.append(
            pd.DataFrame(
                {
                    "epk_id": np.arange(n),
                    "dataset": dataset,
                    "outcome": "Y",
                    "model": model,
                    "seed": seed,
                    "evaluation_split": "test",
                    "T": rng.binomial(1, 0.5, n),
                    "Y": rng.binomial(1, 0.2, n),
                    "cate_pred": rng.normal(size=n),
                }
            )
        )
        rows.append({"dataset": dataset, "model": model, "seed": seed,
                     "qini_auc_normalized": 0.1})
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
    pd.concat(frames, ignore_index=True).to_parquet(
        directory / "predictions.parquet", index=False
    )
    (directory / "run_config.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "seed": seed,
                "outcome_resolved": "Y",
                "max_rows_resolved": None,
                "val_fraction": 0.2,
                "test_fraction": 0.2,
                "evaluation_split": "test",
                "causalpfn_max_context_length": 4096,
                "causalpfn_max_query_length": 4096,
                "causalpfn_num_neighbours": 1024,
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_reuse_finds_a_matching_run(tmp_path):
    from robustness_tests.reuse import find_reusable_run

    run = _fake_completed_run(tmp_path / "retailhero" / "seed_0_20260101_000000")
    found = find_reusable_run(
        tmp_path,
        "retailhero",
        0,
        {"max_rows_resolved": None, "val_fraction": 0.2, "test_fraction": 0.2,
         "evaluation_split": "test", "causalpfn_max_context_length": 4096,
         "causalpfn_max_query_length": 4096, "causalpfn_num_neighbours": 1024},
    )
    assert found == run.resolve()


def test_reuse_refuses_a_run_with_a_different_protocol(tmp_path):
    from robustness_tests.reuse import IncompatibleRunError, find_reusable_run

    _fake_completed_run(tmp_path / "retailhero" / "seed_0_20260101_000000")
    with pytest.raises(IncompatibleRunError, match="causalpfn_max_context_length"):
        find_reusable_run(
            tmp_path,
            "retailhero",
            0,
            {"evaluation_split": "test", "causalpfn_max_context_length": 1024},
        )


def test_reuse_ignores_a_different_seed(tmp_path):
    from robustness_tests.reuse import find_reusable_run

    _fake_completed_run(tmp_path / "retailhero" / "seed_0_20260101_000000")
    assert find_reusable_run(tmp_path, "retailhero", 3, {"evaluation_split": "test"}) is None


def test_ingest_run_tags_provenance_and_axis(tmp_path):
    from robustness_tests.reuse import ingest_run

    run = _fake_completed_run(
        tmp_path / "retailhero" / "seed_0_x", models=("causalpfn", "t_learner")
    )
    metrics, predictions = ingest_run(
        run, ["causalpfn"], {"axis": "scale", "severity_tag": "scale_full", "severity": "full"}
    )
    assert set(metrics["model"]) == {"causalpfn"}
    assert set(predictions["model"]) == {"causalpfn"}
    assert (metrics["axis"] == "scale").all()
    assert metrics["reused_from"].str.contains("seed_0_x").all()
    assert (metrics["status"] == "ok").all()


def test_analysis_consumes_a_completed_sweep(tmp_path):
    _skip_without_sklearn()
    from analyze_robustness import main as analyze_main
    from run_robustness import main as sweep_main

    sweep_main(
        [
            "--datasets", "semisynth_smoke",
            "--axes", "control_share",
            "--severities", "0.5,0.9",
            "--seeds", "0,1",
            "--models", "t_learner,x_learner,dr_learner",
            "--semisynth-covariates", "gaussian:6000x6",
            "--semisynth-base-rate", "0.2",
            "--semisynth-ate", "0.05",
            "--tree-max-iter", "20",
            "--output-root", str(tmp_path),
            "--no-diagnostics",
        ]
    )
    exit_code = analyze_main(
        [
            "--results-root", str(tmp_path),
            "--n-bootstrap", "50",
            "--reference-model", "t_learner",
            "--no-figures",
        ]
    )
    assert exit_code == 0

    analysis = tmp_path / "analysis"
    assert (analysis / "regime_map.md").exists()
    assert (analysis / "bootstrap_per_seed.csv").exists()
    spearman_file = analysis / "spearman_control_share.csv"
    assert spearman_file.exists(), "ground-truth data must produce an agreement table"
    frame = pd.read_csv(spearman_file)
    assert (frame["n_methods"] == 3).all()
