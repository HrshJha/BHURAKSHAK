# BHURAKSHAK optimization task board

Status recorded after the bounded run. Original artifacts and locked corpus preserved.

| ID | Objective | Input → output | Acceptance criteria | Status | Evidence |
|---|---|---|---|---|---|
| A1 | Comparable baseline reproduction | original params + dev → refit baseline | same parameter groups/fit/calibration/decision partitions as candidate | DONE | original_config_refit_folds.json; model_comparison.csv |
| A2 | Probability/schema/threshold audit | source → strict inference and fail-closed features | explicit class order, no argmax bypass, missing inputs rejected | DONE | test_optimization.py; audit.md |
| B1 | Leakage and label audit | v2 provenance + dev metadata → audit | no gated/label features, parameter-group disjointness | DONE | audit.json; fold_assignment.csv; full_tests.log |
| C1 | Generalization diagnostics | dev + old aggregate report → scenario/distribution plots | no locked examples used in development | DONE | scenario_errors.csv; feature_group_distributions.csv; four diagnostic PNGs |
| D1 | XGBoost optimization | 58,644 train windows → 20 trials | fixed seed, 3 grouped folds, pruning, checkpoints | DONE | risk_study.json (17 complete, 3 pruned) |
| D2 | Isolation Forest optimization | healthy inner-fit windows → 20 trials | disjoint threshold/scoring groups, anomaly FPR measured | DONE | if_study.json (11 complete, 9 pruned) |
| E1 | Class weights and thresholds | development inner groups → calibrated gates | jointly constrained Normal FPR; 1/5/10/20% alternatives | DONE | studies; operating_points.json |
| F1 | Feature selection/importance | two allowed feature sets → ablation/importance | grouped comparison; no unavailable modalities | DONE | ablation_displacement_health_folds.json; permutation_importance.csv |
| F2 | Physically consistent expanded training corpus | generator audit → corrected versioned corpus | coupled scenario tilt/strain, observable anomaly injections, domain-approved label semantics | BLOCKED | audit.md; existing data retained, suspect channels quarantined; no unjustified augmentation |
| F3 | Correct missing-endpoint velocity schema | old feature definition → versioned rebuild | first/last finite endpoint time span, all downstream baselines rebuilt | BLOCKED | audit.md; requires separate schema/data rebuild, frozen evaluation preserved |
| G1 | Calibration and alert sweep | separate calibration/decision groups → frozen config | compare methods; measured alert episode/delay tradeoff | DONE | best_development_config.yaml; 12-setting alert_development_sweep.json |
| G2 | Physical lead time and recovery validation | onset/recovery data → validated alert behavior | actual event-onset truth, supported recovery state | BLOCKED | scenario-wide labels and escalation-only engine; alert_validation.json |
| H1 | Robustness and latency | candidate/dev → sensitivity and workstation profile | no retuning; clearly state perturbation and timing scope | DONE | robustness.json; latency.json |
| I1 | Independent final evaluation | pre-frozen candidate → fresh 300-event corpus | one-use lock, same-corpus original/refit/candidate comparison | DONE | independent_test_lock.json; independent_metrics.json; reuse_refusal.log |
| I2 | Acceptance gate | independent results → deployment decision | Critical recall >=89%, Normal FPR <=5% | BLOCKED | candidate 29.22% recall, 7.44% FPR; experimental only |
| J1 | Tests and inference smoke | scripts/bundle → regression evidence | full suite and label-free CLI work | DONE | full_tests.log; 9-row label-free smoke test |
| J2 | Reproduction/reporting | studies/metrics/config → final deliverables | every trial recorded, commands and limitations explicit | DONE | final_report.md; experiment_log.csv; best_development_config.yaml |

No remaining TODO items imply additional unbounded tuning. Binary alternatives, TreeSHAP and new scenario augmentation were deferred under the bounded budget; two feature families and group-block permutation importance were measured. Independent evaluation is consumed and will not be tuned against.
