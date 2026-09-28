# Tuning notes

## Leakage found and fixed

- Group C read `anomaly_label` in both `neighbor_anomaly_fraction` and `hotspot_density`; the old `hotspot_density` had Pearson r **0.9176** and ROC-AUC **0.9663** against `anomaly_label`. Group C now uses only an observable robust-z deformation-rate proxy when a same-time neighbor graph exists. The current corpus has one node per event, so all eight Group C features are NaN and provenance-gated out of XGBoost and Isolation Forest. See `reports/leakage_audit.md`.
- `center_mode="detected"` also used `anomaly_label`, while the feature store and pipeline explicitly requested `oracle`. Center estimation now uses current-snapshot observable displacement only; model builds reject oracle mode. The stored artifact records detected/gated mode.
- Group B persistence and change-point values were computed once from the entire event and copied to earlier windows. They now use only the prefix through each current window; future truncation tests verify the invariant.
- Pipeline neighbor confirmations previously grouped flagged nodes globally by reused node ID. Counts are now restricted to same-event, same-window neighbors, and alert persistence is isolated by event; the current data yields zero valid confirmations.
- Group F's expected deformation comes from fixed `configs/physics.yaml` parameters, not per-event sampled simulator values. It is marked `config_derived`; deployment must provide the same static site parameters.
- DGPS, InSAR, and synthetic terrain features without a deployable source remain gated. DGPS residuals are evaluation-only.

## Tests fixed

- Pinned environment and 17 baseline failures: see `fix_leak_log.md` rows P0-ENV/P1-*.
- Label-blind builds, label permutation, causal window truncation, Group C gating, oracle refusal, provenance coverage, schema drift, and model-input gates are covered by tests.
- Two Group C expectations were updated because they encoded label-derived behavior; the changes are justified against PRD §13's observable feature inputs and the explicit corpus gate. The IF ablation test no longer expects a Group C input because this corpus has no co-temporal neighbors; the §14 spatial arm remains configured but is inactive until a valid multi-node corpus exists.

## Model comparisons (development grouped CV)

| Model | Macro PR-AUC | Critical recall | Macro F1 | Normal false-alarm rate |
|---|---:|---:|---:|---:|
| Threshold rule | 0.3703 ± 0.0021 | 0.0581 ± 0.0127 | 0.4112 ± 0.0046 | 0.2031 ± 0.0013 |
| Logistic C=10 | 0.5568 ± 0.0182 | 0.6396 ± 0.0817 | 0.4855 ± 0.0017 | 0.3846 ± 0.0104 |
| Default XGBoost | 0.9009 ± 0.0083 | 0.6626 ± 0.0548 | 0.8297 ± 0.0085 | 0.0215 ± 0.0016 |
| Tuned XGBoost | 0.8908 ± 0.0064 | 0.8170 ± 0.0471 | 0.7260 ± 0.0064 | 0.1609 ± 0.0050 |

XGBoost passed its logistic false-alarm constraint in every fold and improved Critical recall over the logistic baseline. It does not beat default XGBoost on macro PR-AUC, macro F1, or false-alarm rate. The search was stopped at 16 recorded trials (12 complete, 4 pruned) for turnaround; it did not reach the proposed 200+ trial budget.

Isolation Forest's 100-trial study recorded 41 complete and 59 pruned. Tuned anomaly PR-AUC was 0.5862 ± 0.0150; recall at the healthy-validation 99th percentile was 0.4148 ± 0.0128, healthy false-positive rate 0.0237 ± 0.0012. Training used healthy rows only; contamination was not derived from injected prevalence.

The 60-trial forecaster study had 59 complete and 1 pruned trial. The selected TCN predicts only the next window because each event has nine windows. Repeated grouped CV normalized MSE was 0.00146 ± 0.00023 versus persistence at 0.03728.

## Calibration and thresholds

Isotonic calibration was selected on development validation groups (Brier 0.1151, ECE 0.0089; sigmoid score 0.1591). Configured development operating targets were met: WARNING recall 0.803 / FPR 0.082; CRITICAL recall 0.904 / FPR 0.017. The locked test will provide the single final evaluation and will not be used for retuning.

## Robustness and workstation profile

The shuffled-label control yielded mean macro PR-AUC 0.3344 (chance reference approximately 1/3) and macro F1 0.0321. The learning curve's validation macro PR-AUC rose from 0.904 at 25% of train events to 0.931 at 100%; Group F removal caused the largest feature-group drop. TreeSHAP's top features are `physics_residual`, `physics_residual_velocity`, and `rolling_min`; their provenance is observable/config-derived and documented in the feature registry. The 3 top configurations were repeated over seeds 42, 43, and 44 with event-grouped 3-fold CV. All details, scenario recall, IF ablation, and permutation importance are in `reports/tuning/robustness.json`.

Workstation CPU p50/p95 latency per window: tuned XGBoost 0.317/0.613 ms; Isolation Forest 2.989/3.829 ms; forecaster 0.042/0.045 ms. This is a workstation profile, not device performance. The seeded tabletop stand-in has 14,552 matched windows, 18 absent model inputs, and Critical recall 0.0; physical tabletop validation remains pending.

## Final locked-test result and open items

The frozen synthetic test set was evaluated once after commit `d09db22` and tag `frozen-for-test`; `reports/test_lock.json` is now `evals_run: 1`. On 1,875 events / 16,875 windows, tuned XGBoost achieved macro PR-AUC 0.5029, macro F1 0.4458, Critical recall 0.0933, and Normal false-alarm rate 0.1508. Default XGBoost Critical recall was 0.0457 (FAR 0.0372); logistic Critical recall was 0.7650 with FAR 0.6263. The tuned model did not provide strong holdout Critical recall.

Alert-engine false alarms were 0.046 per normal event day. Median lead time was −3.33 h (P10 −11.67 h), so this holdout does not show useful early warning. The set was not reread and no tuning used its outcomes. G-5 remains unresolved because there is one node per event. Real-mine performance is unvalidated.
