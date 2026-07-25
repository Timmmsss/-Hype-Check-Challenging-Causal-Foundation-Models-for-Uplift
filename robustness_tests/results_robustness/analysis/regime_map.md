# Regime breakdown map

Primary metric: `qini_auc_normalized`. Reference model: `causalpfn`.

Bootstrap intervals resample evaluation rows with replacement, paired across models within a seed and pooled across seeds.

## Where the advantage breaks

| dataset | axis | metric | breaks | breakpoint | difference | interval | worst |
|---|---|---|---|---|---|---|---|
| hillstrom | control_share | qini_auc_normalized | no | - | - | - | -0.2318 at control_share_trate0p700 |
| hillstrom | control_share | uplift_at_10pct | no | - | - | - | -0.0108 at control_share_trate0p700 |
| hillstrom | conversion | qini_auc_normalized | no | - | - | - | -69.8173 at conversion_lam12p00 |
| hillstrom | conversion | uplift_at_10pct | no | - | - | - | -0.0068 at conversion_lam0p00 |
| hillstrom | scale | qini_auc_normalized | no | - | - | - | -0.2653 at scale_rows50000 |
| hillstrom | scale | uplift_at_10pct | no | - | - | - | -0.0092 at scale_rows50000 |
| lzd | control_share | qini_auc_normalized | no | - | - | - | -0.1343 at control_share_trate0p850 |
| lzd | control_share | uplift_at_10pct | no | - | - | - | -0.0323 at control_share_trate0p850 |
| lzd | conversion | qini_auc_normalized | no | - | - | - | -0.1757 at conversion_lam0p00 |
| lzd | conversion | uplift_at_10pct | no | - | - | - | -0.0232 at conversion_lam0p00 |
| lzd | scale | qini_auc_normalized | no | - | - | - | -0.1319 at scale_full |
| lzd | scale | uplift_at_10pct | no | - | - | - | -0.0239 at scale_full |
| retailhero | control_share | qini_auc_normalized | no | - | - | - | -0.0038 at control_share_full |
| retailhero | control_share | uplift_at_10pct | no | - | - | - | -0.0248 at control_share_full |
| retailhero | scale | qini_auc_normalized | no | - | - | - | -0.0185 at scale_rows200000 |
| retailhero | scale | uplift_at_10pct | no | - | - | - | -0.0233 at scale_rows200000 |

## PEHE / Qini ranking agreement (H2)

_No ground-truth data in this sweep; run a `semisynth` dataset._

## Hyperparameter influence on the breakpoint

_No hyperparameter variants in this sweep; pass e.g. `--causalpfn-context-variants 1024,4096` to populate this section._

## Cells analysed

538 (dataset, axis, severity, model) rows.
