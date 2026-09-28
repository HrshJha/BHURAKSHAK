# Audit after leakage remediation

## Status

The v2 training feature matrix is label-blind by construction and checked by label-drop and label-permutation tests. Provenance allow-lists restrict model inputs to observable/config-derived features; unavailable or simulator-only modalities are gated. The old leaky v1 metrics are archived under `reports/superseded_leaky/` and withdrawn.

## Critical/high findings

No known CRITICAL/HIGH leakage finding remains in the audited model-input path. This statement reflects repository tests and provenance inspection, not a guarantee of real-mine validity.

## Open findings

- G-5 remains unresolved: the corpus contains one node per event, so cross-node/neighbor confirmation cannot be validated. Group C spatial fields stay gated.
- Group F uses static config-derived priors; deployment must supply valid site configuration.
- The seeded tabletop transfer fixture is synthetic and feature-scarce; Critical recall was 0.0 with 18 model inputs missing. Physical hardware validation is pending.
- PRD edge latency/memory targets and alert false-alarm/lead-time acceptance thresholds are not numerically defined; report measured values without claiming pass/fail.

## Evidence

See `reports/leakage_audit.md`, `reports/tuning/robustness.json`, `reports/tuning/tuned_cv_metrics.json`, `reports/test_lock.json`, and `reports/final_eval.md`. The final report records the single locked-set evaluation.
