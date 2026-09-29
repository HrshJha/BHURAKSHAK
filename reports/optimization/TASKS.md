# BHURAKSHAK optimization task board

All evidence is synthetic. Existing locked data and original artifacts remain untouched.

| ID | Objective | Input → output | Acceptance criteria | Status | Evidence |
|---|---|---|---|---|---|
| A1 | Reproduce comparable baseline | development features/config → grouped baseline | identical folds and metrics for candidates | IN PROGRESS | optimization runner |
| A2 | Audit inference schema/probabilities | model code → fixes/tests | explicit class order, threshold semantics | IN PROGRESS | source inspection |
| B1 | Audit leakage/labels | provenance/split metadata → audit | no target-derived features or shared regimes across folds | IN PROGRESS | existing v2 gates |
| C1 | Diagnose generalization gap | dev + old aggregate report → diagnostics | scenario errors and group distributions | IN PROGRESS | rapid scenario absent from development |
| D1 | Tune XGBoost | train groups → 20-trial study | seeded bounded Optuna with pruning | TODO | |
| E1 | Tune weights/thresholds | inner calibration/decision groups → operating points | 1/5/10/20% normal FPR tradeoffs, outer fold scoring | TODO | |
| F1 | Test feature ablations | provenance allowlist → selected subset | identical fold comparison, importance | TODO | |
| F2 | Audit synthetic physical consistency | generator → limitations/quarantine | no invented channels or changed labels | IN PROGRESS | source inspection |
| G1 | Calibration and alert sweep | development predictions → calibration/alert reports | calibration separate from threshold selection; neighbor gate preserved | TODO | |
| H1 | Robustness | frozen candidate/development → stress metrics | no test-dependent changes | TODO | |
| I1 | Freeze and independently evaluate | candidate + new seeded synthetic corpus → one-use report | immutable config/artifact hashes before generation | TODO | |
| J1 | Tests and documentation | implementation/results → final report | reproducible commands, honest limitations | TODO | |
