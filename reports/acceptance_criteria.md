# Acceptance criteria status after leakage remediation

| Criterion | Evidence/status |
|---|---|
| Label-blind features and provenance gate | Passing regression tests; v2 model inputs exclude gated spatial, terrain, InSAR, and unavailable modalities. See `reports/leakage_audit.md`. |
| Event-disjoint development evaluation | Three `StratifiedGroupKFold` folds in `reports/tuning/tuned_cv_metrics.json`; no locked test access during tuning. |
| Tuned XGBoost false-alarm constraint | Revalidated against selected C=10 logistic baseline; all three fold FARs passed. Search stopped at 16 actual trials for turnaround, so this is not a 200+ trial search. |
| Isolation Forest healthy-only training | 41 completed / 59 pruned from 100 requested trials; healthy-validation 99th-percentile threshold. |
| Forecaster vs persistence | Selected next-window TCN beat persistence on grouped development CV; longer horizons unsupported by sequence length. |
| Shuffled-label control | Macro PR-AUC 0.3344, near the approximately 1/3 chance reference; macro F1 0.0321. |
| Neighbor confirmation / G-5 | Not verifiable: one node per event. Spatial confirmation is gated; no synthetic cross-event neighbors are fabricated. |
| Alert-engine targets / de-escalation | Locked synthetic test: 0.046 false-alert episodes per normal event day; median lead time −3.33 h and P10 −11.67 h. Neighbor confirmation and de-escalation remain unavailable/unimplemented; PRD targets remain undefined. |
| Physical tabletop / real-mine validation | Physical hardware and real-mine evidence absent. Seeded synthetic tabletop stand-in transfer has Critical recall 0.0 with 18 model features missing; this is not physical validation. |
| Fresh locked-test result | Evaluated exactly once after freeze; `reports/test_lock.json` has `evals_run: 1`. Tuned model Critical recall is 0.0933; no early-warning success claim is supported. |
