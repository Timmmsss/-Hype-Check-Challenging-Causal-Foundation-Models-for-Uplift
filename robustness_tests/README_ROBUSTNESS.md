# Robustness Tests (goal G2): the regime-breakdown map

Maps the three regimes where CausalPFN's uplift-ranking advantage is claimed to
break (hypothesis H3 in `main.tex`): data beyond the context window, a small
control group, and low conversion. Reports Qini and uplift@k with **test-set
bootstrap** confidence intervals over at least ten seeds, PEHE where ground
truth exists, and the Spearman agreement between the PEHE ranking and the Qini
ranking *inside each degraded regime* (H2).

The output — `breakpoint_<axis>.csv` — is the handoff artifact for G3
(CausalPFN-Rank): it states where and how badly the model breaks.

`baseline_benchmark/` is reused and **not modified**. Its data pipeline, split
protocol, metrics and estimators are imported directly; all 14 of its existing
tests still pass unchanged.

## Layout

```
robustness_tests/
├── robustness_tests/
│   ├── axes.py          # the three axes, severity grids, ordering
│   ├── resampling.py    # Keith et al. accept/reject core + the axis rules
│   ├── prepare.py       # split -> resample -> train-only preprocessing
│   ├── diagnostics.py   # Keith §4.4 checks + event-count gating
│   ├── qlearner.py      # Q-Learner       (low-conversion specialist)
│   ├── gpcate.py        # GP-CATE         (small-control specialist)
│   ├── semisynth.py     # binary semi-synthetic RCT with known tau
│   ├── pehe.py          # PEHE, Spearman agreement (numpy-only statistics)
│   ├── bootstrap.py     # paired percentile bootstrap over predictions
│   ├── registry.py      # model factory, delegating to baseline_benchmark
│   └── reuse.py         # ingest completed runs for the undegraded anchor
├── run_robustness.py    # the sweep
├── analyze_robustness.py# the regime map
└── tests/test_robustness.py
```

## Method

Two papers ground the resampling: Gentzel et al. (`gentzel21a.pdf`) and Keith et
al., *RCT Rejection Sampling for Causal Estimation Evaluation*
(`2307.15176v3.pdf`). Keith's Algorithm 1 accepts unit *i* with probability

```
(1 / M) · P*(T = t_i | C_i) / P(T = t_i),     M ≥ sup P*(T|C) / P(T)
```

Their Theorem 3.2 gives identification; their Proposition 3.1 shows why
Gentzel's OSRCT does not — it conditions on selection, leaving
`P(C) ≠ P(C | S = 1)` with no non-parametric functional recovering the original
marginal. **The acceptance rule must never condition on `Y` at the unit level.**

Both shipped axes use *special cases* of that sampler which are strictly safer
than the confounded setting the paper studies:

| Axis | Rule | What survives |
|---|---|---|
| `control_share` | `P*(T=1\|C) = π`, constant in `C` | Acceptance is uniform inside each arm, so `P(C)` and `P(Y\|T,C)` are preserved **and** `T ⟂ C` holds. Still a valid RCT, just with a shifted arm ratio. |
| `conversion` | acceptance is a function of `C` alone | `T ⟂ (C, Y(0), Y(1))` holds, i.e. a valid RCT on a *shifted covariate population*. `P(C)` moves by design; the CATE stays identified with no adjustment. |
| `scale` | uniform subsample | Nothing changes but the sample size. |

The general C-dependent sampler is implemented (`rejection_sample`), so a
genuinely confounded study remains available, but it breaks `T ⟂ C` and so makes
Qini stop being a causal quantity. It requires `--allow-confounding`.

`test_conversion_rule_never_depends_on_the_outcome` encodes the Proposition 3.1
constraint directly: permuting `Y` must leave the accepted row set bit-identical.

### Order of operations

```
load_cleaned_dataset()   # reused verbatim, incl. the global --max-rows cap
_split_indices()         # reused verbatim: stratified / group-safe, unchanged
resample indices         # on the targeted splits only, at raw-index level
equalize rows            # optional, uniform, see below
_validate_splits()       # reused
_make_preprocessor()     # fitted on the RESAMPLED training rows only
```

Resampling *after* the split inherits the split protocol and its group-safety
guarantees untouched, and keeps the preprocessor train-only-fitted.

### Which splits are degraded

| Axis | Default | Why |
|---|---|---|
| `scale` | train + validation | The evaluation split stays byte-identical across severities, so Qini is directly comparable. |
| `control_share` | train + validation | Same, and the evaluation control arm stays large enough to estimate Qini at all. |
| `conversion` | all splits | Low conversion is a property of the deployed population, not only of the training data. |

`--degrade-splits {train,train+val,all}` overrides this; running the opposite
setting is the documented sensitivity check.

## Three things that would silently invalidate results, and how they are handled

1. **The conversion knob thins the sample.** A selection rule that rejects hard
   also shrinks `n`, which would confound the conversion axis with the scale
   axis. `--equalize-rows N` trims every severity to a common training size
   (other splits scaled in the same 60/20/20 proportion) by a *uniform* draw,
   which changes nothing else. **Set it on the conversion and control-share
   axes**; `--dry-run` warns when it is missing. Equalization is a cap, not a
   guarantee — a harsh severity can yield fewer rows than the target, and that
   is reported as `equalization_reached = false` rather than hidden.

2. **In-sample sharpness of the risk model.** Scored in sample, the
   `P(Y=1|C)` model separates training rows far better than held-out rows, so
   the same knob thinned training conversions ~10× while evaluation conversions
   fell only ~2.4× — a property of the model, not of the regime. The training
   rows are therefore scored **out of fold**; the two now move together.

3. **The achievable conversion floor is bounded** by how well the covariates
   predict the outcome out of sample. That is exactly Keith et al.'s
   precondition, which `check_outcome_dependent_covariate` tests and raises on.
   The severity grid is therefore always reported by the **realized** base rate
   (`train_conversion_rate`, `eval_conversion_rate`), never by the knob alone.

## The two 2026 specialists

### Q-Learner (arXiv:2605.26288) — low conversion

`τ_ratio(x) = [p(x)/(1−p(x))] · [(1−e(x))/e(x)]` with `e(x) = P(W=1|x)` and
`p(x) = P(W=1|Y=1,x)`. Two *classifications*, never an outcome regression on a
rare event — which is why it survives low conversion. Clipping follows the
paper's Appendix A (`e, p ∈ [0.01, 0.99]`, ratio ∈ `[0.01, 100]`).

Two registered names, because the scales are not interchangeable:

- `q_learner` — **difference scale** `μ1 − μ0`, recovered with a third
  classifier for `m(x) = P(Y=1|x)` via `μ1 = m·p/e`, `μ0 = m·(1−p)/(1−e)`. The
  default, so it enters both the Qini and the PEHE tables.
- `q_learner_ratio` — the paper's native ratio score. A valid targeting score
  (the paper evaluates it with Qini) but not comparable to a difference-scale
  ground truth, so its PEHE is reported as `NaN`.

It refuses to fit below `--q-min-converters` (default 50) or when converters
appear in only one arm. That guard *will* fire at high conversion severity; the
cell is recorded with `status = error` and the sweep continues, because "not
estimable in this regime" is itself a result.

### GP-CATE (arXiv:2605.27473) — small control arm

Per-arm GP, `τ(x)|D ~ N(m1−m0, s1²+s0²)`, constant×RBF+white kernel with
empirical-Bayes hyperparameters (2 restarts). `predict_cate_interval()` exposes
the credible interval for the coverage work in G3.

**Deviations from a literal reading of the paper**, forced by this repo's data
and to be stated in `main.tex`:

- The paper assumes real-valued `Y` with a Gaussian likelihood; our outcomes are
  binary, so this is a linear-probability-style approximation. Ranking is little
  affected; interval *calibration* is more so, and that matters wherever the
  intervals are used.
- An isotropic length scale over 60+ one-hot columns (LZD) is meaningless, so
  features are projected to `--gp-n-components` principal components first.
- Exact GP inference is cubic, so both arms are capped (`--gp-max-control` 2000,
  `--gp-max-treated` 500). The harder treated cap follows the paper's own
  recommendation: in the few-placebo regime the treated arm contributes only a
  small share of the CATE variance.

## Ground truth

Real campaign data reveal one potential outcome per unit, so PEHE is not
computable there. IHDP and ACIC 2016 have *continuous* outcomes, which the
existing harness rejects (`metrics.py::_as_arrays` requires a binary outcome;
`models.py` fits classifiers).

The primary ground-truth source is therefore a **binary-outcome generator over
real covariates**:

```
μ0(x) = σ(α + z(x))          α solved numerically for the target base rate
τ(x)  = heterogeneous, known
y     ~ Bernoulli(clip(μ0 + T·τ))
```

Both arm probabilities are clipped *before* the draw, so the true CATE is
`p1 − p0` **exactly**. Effect heterogeneity uses a second, differently weighted
covariate index, so ranking by predicted outcome does not trivially solve the
uplift task. Covariates come from `gaussian:<n>x<d>` (self-contained),
`ihdp:<path to npz>` (the real 25 IHDP covariates), or any cleaned dataset name.

Every existing estimator and metric works on it unchanged, and it accepts the
same axis knobs, so all three regimes are swept **with** ground truth — which is
what a per-regime Spearman table requires.

*Not yet built:* real IHDP/ACIC with their original continuous outcomes. That
needs a parallel regression model path plus an additive
`_as_arrays(..., require_binary_outcome=False)` flag in `metrics.py`. The flag
was left out deliberately rather than added as dead code, so `baseline_benchmark`
is currently untouched.

## Usage

```bash
cd robustness_tests

# 1. Preview the matrix and the cost. Always do this first.
python run_robustness.py --datasets criteo,hillstrom,lzd,retailhero \
    --axes all --seeds 0,1,2,3,4,5,6,7,8,9 --dry-run

# 2. Sweep, with hyperparameters frozen from the seed-42 tuning batch and the
#    completed full-table runs reused for the undegraded anchor.
python run_robustness.py --datasets retailhero --axes all \
    --seeds 0,1,2,3,4,5,6,7,8,9 --max-rows 0 \
    --frozen-params <tuning_batch>/results/retailhero/tuning_seed_42/best_params.json \
    --reuse-existing ../baseline_benchmark/results \
    --equalize-rows 40000

# 3. Semi-synthetic, for PEHE and the H2 agreement table.
python run_robustness.py --datasets semisynth_gauss --axes all \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --semisynth-covariates gaussian:200000x12 --semisynth-base-rate 0.05

# 4. The regime map.
python analyze_robustness.py --results-root results_robustness --n-bootstrap 1000
```

`--reuse-existing` ingests completed runs for the **anchor severity only**,
after checking `run_config.json` on outcome, `max_rows`, split fractions,
evaluation split and the CausalPFN context settings. A mismatch is refused
loudly, never relabelled. When every requested model is already covered, the
cell needs no data preparation at all — that is the case that saves the
full-table CausalPFN hours. When only some are covered, the rest are fitted and
the evaluation row identifiers are asserted to match before the two are mixed.

Hyperparameters are **frozen across severities**. Re-tuning per severity would
confound the axis with the search budget.

`analyze_robustness.py` also runs standalone against the existing part-1/2
results, which is how the missing test-set bootstrap intervals for the current
main table are produced:

```bash
python analyze_robustness.py --results-root ../baseline_benchmark/results/retailhero \
    --reference-model x_learner --no-figures
```

## Output

```
results_robustness/<dataset>/<axis>/<severity_tag>/seed_<seed>_<timestamp>/
├── metrics.csv            # baseline columns + axis, severity, severity_tag,
│                          #   realized rates, n_train, degenerate, status,
│                          #   equalization_reached, pehe, abs_ate_error
├── predictions.parquet    # + axis, severity_tag, tau_true when known
├── resample_manifest.json # ResampleRecord per split + all diagnostics
├── data_manifest.json, run_config.json, splits.parquet, preprocessor.joblib
```

Analysis writes `qini_bootstrap_<axis>_<metric>.csv`, `paired_<axis>_<metric>.csv`,
`breakpoint_<axis>.csv`, `pehe_<axis>.csv`, `spearman_<axis>.csv`,
`regime_map.md` and one figure per axis (matplotlib optional).

The bootstrap resamples evaluation rows with replacement, **paired across models
within a seed** — so `CausalPFN − best alternative` gets a valid paired interval,
which is the operational form of H3 — then pools per-seed draws so the interval
carries both row and split noise.

## Tests

```bash
python -m pytest tests/test_robustness.py -q        # 63 tests, synthetic data only
python -m pytest ../baseline_benchmark/tests/test_smoke.py -q   # 14, must stay green
```

No access to the cleaned campaign data is required. Tests needing sklearn skip
cleanly without it.
