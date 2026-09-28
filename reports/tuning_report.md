# Hyperparameter tuning report

## Scope and decision

The stored spatial features contain ground-truth `anomaly_label` information. XGBoost and Isolation Forest tuning was stopped before any fit or held-out scoring, as required for a leakage finding. The only model tuned was the physical forecaster, whose input/output path does not use the risk or anomaly labels.

The existing held-out test groups had already been scored by prior repository artifacts, so they were treated as burned and were not used for tuning or rescored here. No fresh test-set claim is made.

## Forecaster grouped search

Eight seeded candidates were sampled from the recorded search space. Selection used 3-fold `GroupKFold` by `event_id` on the 8,125 development events (train and validation assignments only). The top three configurations were rerun over the same group folds with seeds 42, 43 and 44. The held-out test events were excluded from the tuning dataframe.

| Model / baseline | Architecture | Width | History | Horizon | Grouped CV normalized MSE |
|---|---:|---:|---:|---:|---:|
| Tuned forecaster | LSTM | 16 | 3 | next window (1 step) | **0.02735** selection; **0.02810 ± 0.00343** across three seeds |
| Persistence | — | — | — | next window (1 step) | **0.03720** |

All three repeated seeds beat persistence. Fold/channel MAE, RMSE, and raw trial values are recorded in `reports/tuning/forecaster_study.json` and `reports/tuning/forecaster_trials.csv`. The configuration is frozen in `configs/model_params.yaml`; the development-fitted artifact is `models/temporal_model/forecaster_tuned.pt` and records model, schema, and dataset versions.

Every sequence has exactly nine windows. With the selected three-step history, the data supports a one-step target; the longer configured horizons produce no examples and were not claimed or tuned. The forecaster is not calibrated because its outputs are continuous physical quantities, not class probabilities.

## Risk classifier and anomaly detector

| Model | Baseline vs default vs tuned | PR-AUC / recall / F1 / calibration | Status |
|---|---|---|---|
| XGBoost | Not compared on this pass | Not measured | Blocked by target-derived Group C inputs |
| Isolation Forest | Not compared on this pass | Not measured | Blocked by target-derived Group C inputs |

Thus no claim is made that tuned XGBoost beats threshold-rule or logistic regression. No anomaly-score operating point, class threshold, calibration choice, permutation importance, SHAP result, shuffled-label control, or alert-engine metric was generated.

## Test-set line

**Not run.** The repository’s pre-existing held-out test outputs mean that split is burned. This pass used development-only grouped CV for the forecaster and did not touch test rows.

## Edge profile and checks

On this workstation, forecaster CPU inference for one three-step history measured **0.0298 ms p50 / 0.0312 ms p95** over 1,000 calls after warmup. Process peak RSS was **206.4 MB** including Python and PyTorch imports; the artifact is **9.8 KB**. This is not an edge-device measurement, and the PRD does not provide a numeric NFR-2 budget.

The requested full test suite was run and has **17 failures**: 6 Isolation Forest tests and 10 packet-validation tests fail on mutation of read-only NumPy views; one Group G test finds a 1,000× velocity unit-scale mismatch. Details and required follow-up are in `tuning_notes.md`. No tests were modified.
