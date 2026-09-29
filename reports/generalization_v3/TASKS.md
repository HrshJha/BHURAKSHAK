# V3 task disposition

| Task | Outcome | Evidence |
|---|---|---|
| Diagnose label/observability gap | DONE; original scenario identity and actual local severity are distinct targets | root_cause.json; retained legacy labels; identical-observation regression test |
| Shared displacement/tilt/strain physics | DONE for versioned synthetic generator | coupled_v3.py; analytic/finite-difference and strain-geometry tests |
| Observable vibration/disturbance | DONE for synthetic signal generation; field coupling remains unvalidated | sampled waveform/pulse tests |
| Correct velocity and fault cadence | DONE in v3; explicit v2 replay retained | finite-endpoint and 10-minute drift regression tests |
| Data generation gate | DONE; physical tests passed before final retraining | physics_tests.log |
| Baselines and bounded optimization | DONE; 20 corrected-data trials, 17 complete, 3 pruned; three ML families, fixed rule, ablations, original-settings refit | study.json; development_comparison.csv |
| Calibration, weights, causal aggregation | DONE; grouped fit/calibration/decision separation, isotonic calibration, two-stage weights, EWMA | selected_config.yaml; source freeze |
| Development margin and uncertainty | DONE; >92% recall / <3% FPR and group-CI gate before fresh test | validation.json; freeze.json |
| Independent local-severity acceptance | PASSED for explicitly versioned 15/35 mm policy; point estimates and group bounds pass | independent_results.json; independent_lock.json |
| Original scenario-label acceptance | NOT MET; 41.13% recall for v3 candidate under retained labels | separate legacy-target diagnostic; no same-task success claimed |
| Inference and regression verification | DONE; 675 tests passed, label-free raw-observation CLI smoke, one-use refusal | full_tests.log; reuse_refusal.log |
| Mine-safety/early-warning readiness | NOT VALIDATED; long warmup/detection delay, weak Warning recall, prototype-only thresholds | final_report.md; events and calibration metrics |

No physical measurements or additional hardware were invented. Existing deployed artifacts remain unchanged. The first v3 draft was rejected for a cadence bug before independent testing and preserved separately. No tuning followed the final consumed test.
