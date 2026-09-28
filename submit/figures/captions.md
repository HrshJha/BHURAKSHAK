# Figure captions and sources

## fig_01_ml_flow

The pipeline diagram names the eight inference stages; spatial fusion is marked gated because this corpus has no co-temporal neighbors.

Origin: synthetic corpus. Source: `src/pipeline.py; reports/acceptance_criteria.md`.

## fig_02_physics

Config-driven synthetic subsidence profile, temporal growth, and analytic tilt gradient.

Origin: synthetic corpus. Source: `src/simulator/deformation_field.py; src/simulator/temporal_model.py; configs/physics.yaml`.

## fig_03_dataset

Post-fix generated counts and fresh locked-test class support.

Origin: synthetic corpus. Source: `data/synthetic/dataset_manifest.json; reports/final_eval.json`.

## fig_04_test_comparison

Locked synthetic test baseline comparison; alert false-alert/day is separately reported in tables/alert.csv.

Origin: synthetic corpus. Source: `reports/final_eval.json`.

## fig_05_per_class

Locked test per-class metrics and tuned-model confusion matrix.

Origin: synthetic corpus. Source: `reports/final_eval.json`.

## fig_06_calibration

Saved reliability curve from post-fix development validation; not a locked-test calibration curve.

Origin: synthetic corpus. Source: `reports/tuning/reliability.png; reports/tuning_report.md`.

## fig_07_fault_categories

Class recall by selected sensor-fault/noise categories on development validation; false-Critical rates by fault were not retained, so no such claim is made.

Origin: synthetic corpus. Source: `reports/tuning/robustness.json: required_scenario_category_recall`.

## fig_08_forecast

Next-window TCN versus persistence normalized MSE on grouped development CV; longer horizons are unsupported.

Origin: synthetic corpus. Source: `reports/tuning/forecaster_study.json: best`.

## fig_09_explainability

Global mean absolute TreeSHAP for three leading features; a single-alert payload was not retained.

Origin: synthetic corpus. Source: `reports/tuning/robustness.json: tree_shap`.

## fig_10_ablation

Macro PR-AUC after removing active feature groups on development validation.

Origin: synthetic corpus. Source: `reports/tuning/robustness.json: feature_group_ablations`.

## fig_11_edge

Single-window workstation CPU latency and sampled process RSS; no device measurement.

Origin: synthetic corpus. Source: `reports/tuning/robustness.json: workstation_edge_profile`.

## Dropped figures

- Event timeline: the final report does not retain row-level holdout probabilities/timestamps; direct access to the one-use holdout is prohibited after evaluation.
- Isolation Forest score distribution: source reports retain aggregate metrics and threshold, not score samples; no regenerated distribution is claimed.
- Per-alert SHAP payload: only model-level TreeSHAP ranking is retained.
- Physical tabletop trace: available fixture is a seeded synthetic stand-in, not a physical rig; no real-rig figure is included.
