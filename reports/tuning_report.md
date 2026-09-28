# Hyperparameter tuning report

## Current status

Leakage remediation is complete on `fix-leak-retune`. The v1 feature store and every downstream risk-model, ablation, domain-gap, legacy test, and calibration result that used it are archived under `reports/superseded_leaky/`. They are invalid as model results.

The v2 feature store is label-blind. Group C spatial-context values are gated on the current one-node-per-event corpus. The threshold, logistic, and default XGBoost baselines use the same three grouped development folds. Isolation Forest's 100-trial study completed 41 trials and pruned 59. The 60-trial forecaster study completed 59 trials and pruned one; its best next-window TCN has grouped-CV normalized MSE 0.00149 versus 0.03728 for persistence. Repeated grouped CV for the selected configuration (three folds × three seeds) averaged 0.00146 ± 0.00023. XGBoost's 200-trial study is still running. Final selected-model CV, calibration, robustness, and workstation profiling remain pending.

The fresh locked test set is frozen with `evals_run: 0`; it has not been evaluated. `reports/final_eval.md` will be written only after development-only tuning, calibration, robustness, and the model freeze are complete. `scripts/final_eval.py` is the sole authorized reader and must run exactly once.

## Reproducibility and evidence

- Current source feature artifact: `data/features/features_v2.parquet`.
- Feature provenance: `configs/feature_provenance.yaml`.
- Leakage findings and per-feature univariate scan: `reports/leakage_audit.md`.
- Prior contaminated results: `reports/superseded_leaky/` (`superseded_leaky`).
- Test-set lock: `reports/test_lock.json`, currently `evals_run: 0`; access is guarded so only `scripts/final_eval.py` can read the frozen corpus.

Only synthetic-corpus results will be reported. The IF study improves development-fold average precision from 0.5561 to 0.6005, with recall at the validation healthy 99th-percentile threshold increasing from 0.254 to 0.417. The forecaster's selected next-window configuration beats persistence on the same development groups. These are tuning results, not held-out evidence. XGBoost tuning and the locked evaluation will determine the final report; no real-mine performance claim is supported.
