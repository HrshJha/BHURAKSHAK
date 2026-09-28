# ML evidence summary

## Problem
- Estimate movement-risk classes and detect anomalous sensor patterns from generated subsidence signals.
- Evidence is synthetic; it does not establish real-mine performance.

## Data
- Physics-coupled synthetic generator; one event per node, with 60-step windows and 10-step stride.
- Regime-held-out split; grouped CV keeps events together. The fresh locked synthetic test was evaluated once.
- Counts and class support: `tables/dataset.md`; construction overview: `figures/fig_03_dataset.png`.

## Models
- Isolation Forest: unsupervised anomaly score, fit on healthy training rows.
- XGBoost: three-class risk score, compared with logistic regression and a threshold rule.
- Physics check: compares observed movement with config-derived expected deformation.
- TCN forecaster: next-window displacement/tilt estimate, compared with persistence.
- Alert engine: maps risk scores to alert states; current single-node corpus cannot validate neighbor confirmation.

## Results
- Locked synthetic test results: `tables/results.md` and `tables/per_class.md`; confusion matrix: `figures/fig_05_per_class.png`.
- Tuned XGBoost locked Critical recall is low; its score is not an early-warning success claim. Alert lead time is negative in the locked test (`tables/alert.md`).
- Grouped development CV is reported separately in `tables/baselines.md`; development is not the locked test.

## Baselines
- On the locked test, tuned XGBoost has higher Critical recall than default XGBoost but low absolute recall; logistic has higher Critical recall and many more Normal false alarms. Full values are in `tables/results.md`.
- The next-window TCN beats persistence on grouped CV (`tables/forecast.md`).

## Fault checks and workstation profile
- Selected fault-category class recall is in `tables/fault_scenarios.md`; Critical recall is zero in those validation categories. Per-category false-Critical rate was not retained.
- Timing and memory are workstation CPU measurements only (`tables/edge_profile.md`); no device claim is made.

## Limitations
- Synthetic corpus only; no real-mine validation. One node per event gates spatial neighbor confirmation.
- Forecast support is limited to the next window. The tabletop fixture is synthetic, not physical-rig evidence.
- No row-level locked probabilities were retained; no event timeline is shown.

## Not built
- This bundle does not claim field deployment, physical-rig validation, or operational benefit.
