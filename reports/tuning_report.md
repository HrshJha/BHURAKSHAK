# Hyperparameter tuning report

## Development-only results

All figures here are from the leakage-controlled synthetic development corpus and grouped cross-validation. The locked test set remains unevaluated until the final freeze.

| Model | Macro PR-AUC (mean ± SD) | Critical recall (mean ± SD) | Macro F1 (mean ± SD) | Normal false-alarm rate (mean ± SD) |
|---|---:|---:|---:|---:|
| Threshold rule | 0.3703 ± 0.0021 | 0.0581 ± 0.0127 | 0.4112 ± 0.0046 | 0.2031 ± 0.0013 |
| Logistic, C=10 | 0.5568 ± 0.0182 | 0.6396 ± 0.0817 | 0.4855 ± 0.0017 | 0.3846 ± 0.0104 |
| Default XGBoost | 0.9009 ± 0.0083 | 0.6626 ± 0.0548 | 0.8297 ± 0.0085 | 0.0215 ± 0.0016 |
| Tuned XGBoost | 0.8908 ± 0.0064 | 0.8170 ± 0.0471 | 0.7260 ± 0.0064 | 0.1609 ± 0.0050 |

The tuned XGBoost candidate improved Critical recall and macro F1 over the selected logistic baseline while reducing normal false alarms in every revalidation fold. Compared with default XGBoost, tuning raised Critical recall but lowered macro PR-AUC and macro F1 and increased false alarms; tuning is therefore not a uniform improvement over the default model. The model remains constrained by the selected logistic comparator, not by the much lower default-XGBoost false-alarm rate.

The logistic baseline selected C=10 from {0.1, 1, 10} on the same grouped development folds. The XGBoost Optuna run was stopped early for the requested turnaround: 16 actual trials, 12 complete, 4 pruned, 0 failed, seed 42. The best composite objective was 0.8358. Its requested count was 40; do not interpret this short search as the 200+ trial search proposed in the tuning plan.

Isolation Forest used healthy training rows only and a threshold from the validation healthy-score 99th percentile. The 100-trial study produced 41 completed and 59 pruned trials. Its tuned grouped-CV anomaly PR-AUC was 0.5862 ± 0.0150, and anomaly recall at that validation threshold was 0.4148 ± 0.0128 with healthy false-positive rate 0.0237 ± 0.0012.

The forecaster study ran 60 trials (59 complete, 1 pruned). The selected next-window TCN had repeated grouped-CV normalized MSE 0.00146 ± 0.00023 versus persistence at 0.03728. Longer horizons are unsupported by the nine-window event sequences.

## Calibration and operating point

The frozen XGBoost model selected isotonic calibration on a held-out portion of development validation groups (Brier 0.1151, ECE 0.0089; sigmoid score 0.1591). Validation thresholds met the configured targets: WARNING recall 0.803 and false-positive rate 0.082; CRITICAL recall 0.904 and false-positive rate 0.017. These are development operating points, not locked-test results.

## Locked test

The fresh synthetic locked set is read only by `scripts/final_eval.py`, which consumes its single evaluation allowance. The final table and alert-engine figures will be added here after that one run. No real-mine performance claim is supported.

## Artifacts

- Detailed folds: `reports/tuning/tuned_cv_metrics.json`
- Search trials: `reports/tuning/xgboost_study.json`, `reports/tuning/isolation_forest_study.json`, `reports/tuning/forecaster_study.json`
- Frozen development settings and artifact metadata: `configs/model_params.yaml`, `models/tuned/`
- Leakage audit and v1 withdrawn metrics: `reports/leakage_audit.md`, `reports/superseded_leaky/`
