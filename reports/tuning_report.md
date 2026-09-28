# Hyperparameter tuning report

## Current status

Leakage remediation is in progress on `fix-leak-retune`. The v1 feature store and every downstream risk-model, ablation, domain-gap, test, and calibration result that used it are archived under `reports/superseded_leaky/`. They are invalid as model results.

The v2 feature store is label-blind. Group C spatial-context values are gated on the current one-node-per-event corpus. XGBoost and Isolation Forest have not yet been retuned on v2. The previous forecaster artifact was trained under the old schema and is archived pending a new grouped Optuna study.

No fresh locked test set has been evaluated. `reports/final_eval.md` will be written only after the development-only baselines, grouped tuning, calibration, and robustness checks are complete.

## Reproducibility and evidence

- Current source feature artifact: `data/features/features_v2.parquet`.
- Feature provenance: `configs/feature_provenance.yaml`.
- Leakage findings and per-feature univariate scan: `reports/leakage_audit.md`.
- Prior contaminated results: `reports/superseded_leaky/` (`superseded_leaky`).
- Test-set lock: not yet created; Phase 3 must create it with `evals_run: 0` and an access guard before tuning proceeds.

Only synthetic-corpus results will be reported. No model-quality claim is active at this stage.
