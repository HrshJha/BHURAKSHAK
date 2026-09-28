# Tuning notes

## Leakage or data bugs found and where

- **Risk classifier and Isolation Forest stopped before tuning.** `src/features/group_c_spatial.py::emit_group_c` reads ground-truth `anomaly_label` to emit `hotspot_density` (and `neighbor_anomaly_fraction`). `src/features/build_feature_store.py::build_feature_store` defaults to `center_mode="oracle"`, and `src/pipeline.py` explicitly requests oracle mode. In the persisted 90,000-row store, `hotspot_density` has correlation **0.918** with `anomaly_label`; the feature is also consumed by both the XGBoost feature resolver and the Isolation Forest spatial feature set. On this single-node-per-event corpus, `neighbor_anomaly_fraction` is constant, while hotspot density directly exposes the anomaly label. The configured IF healthy mask also selects rows with anomaly label zero, making this feature a train/evaluation shortcut. No XGBoost or IF search, threshold tuning, calibration, or new test scoring was run.
- `oracle` center uses known panel geometry. That is available in this synthetic simulator but does not establish inference-time availability; treat it as a train/serve parity issue in addition to the direct label-derived feature leak.
- The current persisted test groups were already scored by existing experiments and notebooks before this pass. They are **burned** for a new final claim. This pass did not read them for tuning or score them again.
- Full `pytest tests/ -q --tb=no` run: **17 failures**. Six are `tests/test_isolation_forest.py` failures caused by `healthy_baseline_mask` mutating a read-only NumPy view; ten `tests/test_validation.py` failures share a read-only NumPy flag-array mutation in `src/preprocessing/validation.py`; one `tests/test_group_g.py::test_velocity_and_acceleration_are_point_trend_terms` reports velocity 1000× the expected unit scale. These issues predate the tuning artifacts and need correction before relying on those paths. No tests were changed or weakened.

## Whether tuned beats each baseline (yes/no per metric)

- **Forecaster vs persistence:** Yes on normalized MSE. The winning LSTM (width 16, history 3, depth 1, learning rate 0.003; next-window horizon) scored **0.02735** mean grouped-CV normalized MSE vs **0.03720** for persistence. Across seeds 42/43/44, mean ± sample SD was **0.02810 ± 0.00343**, and each seed beat persistence. Full channel MAE/RMSE and fold records are in `reports/tuning/forecaster_study.json`.
- **XGBoost vs threshold rule / logistic regression / untuned XGBoost:** Not evaluated in this pass. Target-derived features block a trustworthy comparison. Existing `experiments/xgboost_vs_baselines.csv` uses a prior split/protocol and is not substituted as a tuning baseline.
- **Isolation Forest:** Not evaluated; the same feature leak blocks trustworthy ROC-AUC, PR-AUC, and fixed-FPR recall comparisons.
- **Forecaster cross-domain/tabletop:** Not evaluated; existing domain-gap artifacts predate this tuning and are not fresh held-out evidence.

## Test-set numbers

- **No new test-set results.** Test groups were not touched because existing artifacts show they had already been scored; the test is burned. Risk-model and IF test metrics are therefore unavailable for this tuning pass. The forecaster result above is grouped cross-validation over development events only, not a test result.

## Operating points chosen and why

- No risk-model class thresholds, anomaly-score thresholds, calibration method, or alert-engine operating point was changed. Tuning those choices before removing label-derived inputs would optimize the leakage path.
- Forecaster target is the next window only: all event/node sequences have exactly nine windows, so longer configured horizons have no valid sequence samples. Training and scoring are grouped by `event_id`; no window from an event crosses a fold.

## Edge latency/RSS

- Tuned forecaster, CPU, one three-step history, 100 warmups + 1,000 timed forward passes: **p50 0.0298 ms, p95 0.0312 ms**. The process peak RSS was **206.4 MB**, including Python/PyTorch imports. Artifact size is **9.8 KB**. These are workstation measurements; the PRD does not state a numeric edge budget and this is not a Raspberry Pi measurement.
- XGBoost/IF edge measurements were not produced because those models were not tuned or refit.

## Things that did not help

- GRU candidates, the tested LSTM width-8 candidate, and most TCN candidates did not beat persistence on the first grouped-CV search. The winning LSTM candidate was the only one advanced to repeated-seed CV.
- A longer search was not run after selecting the LSTM: the repeated-seed winner remained better than persistence on all three seeds. The search was eight seeded random candidates, not a 100/200-trial Optuna study; those requested classifier/IF studies were blocked by leakage.

## Open items

- Remove label-derived anomaly information from inference features and rebuild the feature store without changing labels; add an explicit leakage assertion that checks every model input against label provenance.
- Rebuild clean grouped train/validation/test assignments and feature artifacts after the feature fix. The current test is burned, so use a newly generated/previously untouched held-out regime for a final claim.
- Repair the read-only mask/flag mutations and the velocity unit-scale failure; rerun the full suite before model comparisons.
- Then tune XGBoost, IF, calibration, thresholds and alert-engine behavior on grouped folds; compare against threshold-rule and logistic baselines. Run the requested shuffled-label, feature-ablation, fault/noise, importance/SHAP, and fresh held-out checks.
- The tuned forecaster artifact is development-fitted only and supports only next-window prediction on this corpus. Do not interpret it as validated multi-horizon or real-mine forecasting.
