# BHURAKSHAK comparison-table row: ML evidence

## Step 1 — truth check

- **Correction:** “trained on public …” is unsupported; the primary XGBoost risk model trains on the simulator-generated training split (`scripts/generate_synthetic_nodes.py`, `src/pipeline.py`).
- **Correction:** LoRa nodes, gateway, and OTA are not ML-workstream deliverables; `README.md` and `TASKS.md` place them in separate hardware workstreams.
- **Correction:** the tabletop corpus is a seeded synthetic stand-in, not a physical-rig log; “pilot” or physical validation is unsupported (`data/recorded/tabletop/README.md`).
- Primary classifier: simulator-generated synthetic training split, generator seed `42`; optional InSAR is gated and its workflow uses a synthetic SLC stack (`scripts/generate_synthetic_nodes.py`, `src/pipeline.py`, `reports/acceptance_criteria.md`, `docs/insar_workflow.md`).
- Model chain: XGBoost classes NORMAL/WARNING/CRITICAL; Isolation Forest anomaly score; physics-consistency check; persistent alert ladder GREEN/WATCH/WARNING/CRITICAL (`src/risk/xgboost_model.py`, `src/anomaly/isolation_forest.py`, `src/physics/consistency.py`, `src/risk/alert_engine.py`).
- Leakage gates are complete: `evals_run: 1`, label-blind test passes, and full pytest logs are green (`reports/test_lock.json`, `tests/test_build_feature_store.py`, `reports/pytest_phase8_pretest.txt`).
- Locked test is an unseen scenario-regime synthetic holdout; XGBoost CRITICAL recall is **9.3%** (`data/synthetic/dataset_manifest.json`, `scripts/make_split_assignment.py`, `reports/final_eval.md`).
- The frozen model's tabletop-stand-in transfer had **0% CRITICAL recall**; physical tabletop validation remains pending (`reports/domain_gap.md`, `experiments/domain_gap_stage2.json`).
- Continuous sensing is a design target; the ML repo does not prove a deployed sensor mesh (`prd.md`, `README.md`).
- The repo documents a 12-day Sentinel-1 repeat, not the full 6–12-day range; that range is retained below only as the literature gap supplied in the team's Raniganj row (`docs/insar_study_region.md`; source slide row not in repo).

## Step 2 — replacement cell options

Character counts include spaces and punctuation, not the surrounding quotes. Each option is at most 75 characters.

### Col 2 — Reported concern

| Option | Cell text | Chars |
|---|---|---:|
| A — Conservative | Subsidence-risk classes from physics-simulated sensor data | 58 |
| B — Balanced **(recommended)** | XGBoost NORMAL/WARNING/CRITICAL classes; trained on synthetic sensor data | 73 |
| C — Strongest defensible | XGBoost NORMAL/WARNING/CRITICAL; trained on simulated sensor data | 65 |

### Col 3 — Current monitoring

| Option | Cell text | Chars |
|---|---|---:|
| A — Conservative | ML risk scoring targets the reported 6–12-day satellite revisit gap | 67 |
| B — Balanced **(recommended)** | Planned continuous sensing + ML; literature 6–12-day Sentinel-1 gap target | 74 |
| C — Strongest defensible | Designed continuous sensors + risk ML target the literature 6–12-day gap | 72 |

### Col 4 — Source / basis

| Option | Cell text | Chars |
|---|---|---:|
| A — Conservative | Synthetic unseen-regime test complete; physical rig test pending | 64 |
| B — Balanced **(recommended)** | Synthetic unseen-regime test: CRITICAL recall 9.3%; rig test pending | 68 |
| C — Strongest defensible | Synthetic unseen-regime CRITICAL recall 9.3%; no physical rig test | 66 |

## Recommended row

| Reported concern | Current monitoring | Source / basis |
|---|---|---|
| XGBoost NORMAL/WARNING/CRITICAL classes; trained on synthetic sensor data | Planned continuous sensing + ML; literature 6–12-day Sentinel-1 gap target | Synthetic unseen-regime test: CRITICAL recall 9.3%; rig test pending |

## Evidence for recommended text

| Fact | Value | Source file | Data origin | Split |
|---|---|---|---|---|
| Primary risk-model classes and training source | XGBoost; NORMAL/WARNING/CRITICAL; simulator-generated inputs | `src/risk/xgboost_model.py`; `src/pipeline.py`; `scripts/generate_synthetic_nodes.py` | Synthetic | Train |
| Continuous sensing is a plan, not a claimed deployment | Planned sensing; hardware scope is separate | `prd.md`; `README.md`; `TASKS.md` | Design | N/A |
| Satellite revisit-gap target | 6–12 days from supplied Raniganj row; repo independently documents a 12-day Sentinel-1 repeat | Team table context (Raniganj row, supplied by user; not in repo); `docs/insar_study_region.md` | Literature | N/A |
| Leakage-safe evaluation gates | One locked evaluation; label-blind test and full pytest pass | `reports/test_lock.json`; `tests/test_build_feature_store.py`; `reports/pytest_phase8_pretest.txt` | Synthetic | Test gate |
| Locked XGBoost CRITICAL recall | 9.3% on unseen scenario-regime holdout | `reports/final_eval.md`; `data/synthetic/dataset_manifest.json`; `scripts/make_split_assignment.py` | Synthetic | Unseen-regime test |
| Physical rig status | Pending; current transfer result uses a generated stand-in | `reports/domain_gap.md`; `experiments/domain_gap_stage2.json`; `data/recorded/tabletop/README.md` | Synthetic tabletop stand-in | Trial holdout |
