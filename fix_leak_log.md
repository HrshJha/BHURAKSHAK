# Fix-leak log

| ID | Root cause | Files changed | Test / evidence | Before → after |
|---|---|---|---|---|
| P0-ENV | Test run used Python 3.14.3 and packages that drifted from exact requirements pins. This exposed unsupported assumptions about writable NumPy views and datetime tick resolution; no source commit between `pre-fixall` and `pre-leakfix` changed the affected files. | `.python-version`, `reports/env_before.txt`, `reports/env_pinned.txt`, `reports/env_drift.md` | Clean Python 3.12.13 venv installed from `requirements.txt`; full baseline suite | Python 3.14 baseline: 17 failures → pinned baseline: 585 passed |
| P1-IF | `healthy_baseline_mask` performed in-place `&=` on read-only Series views. | `src/anomaly/isolation_forest.py`, `tests/test_isolation_forest.py` | Read-only `Series.to_numpy` monkeypatch and Arrow-backed CoW regression; full suite | 6 failing IF tests → full suite 590 passed |
| P1-DQ | `validate_packets` performed in-place `|=` on a read-only result from pandas/NumPy. | `src/preprocessing/validation.py`, `tests/test_validation.py` | Read-only `Series.to_numpy` monkeypatch and Arrow-backed CoW regression; full suite | 10 failing packet-validation tests → full suite 590 passed |
| P1-UNIT | Group G converted raw datetime ticks as nanoseconds although newer pandas may use microseconds. The feature contract is linear trend in mm/day converted to mm/year; acceleration is mm/year² (PRD §13 Group G and module contract). | `src/features/group_g_dgps.py`, `tests/test_group_g.py` | Explicit datetime64[us] public-path regression; expected CP0 trend −136.97 mm/year | Python 3.14 produced −136,968.75 mm/year → normalized timestamp unit gives −136.97 mm/year |
| P2-C-LABEL | Group C `hotspot_density` and `neighbor_anomaly_fraction` read `anomaly_label`; detected-center mode also derived its centroid from anomaly labels. The production corpus has one node per event, so spatial context cannot be computed honestly. | `src/features/group_c_spatial.py`, `src/features/build_feature_store.py`, `src/pipeline.py`, `configs/features.yaml`, `configs/feature_provenance.yaml`, `configs/feature_schema_v2.yaml` | Label-blind build, permuted-label invariance, single-node gate, oracle refusal; old/new univariate scan in `reports/leakage_audit.md` | v1 `hotspot_density`: r=0.9176 and anomaly AUC=0.9663 → v2 all 90,000 rows gated NaN; model allow-list excludes all Group C inputs |
| P2-B-FUTURE | Group B persistence and change-point score were whole-series scalars copied to every window, so early windows depended on future data. | `src/features/group_b_temporal.py`, `tests/test_group_b.py`, `tests/test_build_feature_store.py` | Prefix truncation tests on Group B and the full feature store | Earlier features could change when future windows were present → all windows computed from a causal prefix; equality holds under future truncation |
| P2-PIPELINE-SPATIAL | Pipeline confirmations aggregated flags by node ID over unrelated events; alert persistence also reused node state across events. | `src/pipeline.py`, `tests/test_pipeline_spatial.py` | Separate-event nearby-node test plus same-event/same-window positive test | Independent events could confirm and advance one another → confirmations require co-temporal same-event neighbors; alert keys include event_id; current one-node corpus yields zero |
| P2-PROVENANCE | Model resolvers had no executable source gate; XGBoost and IF could include leaky or unavailable spatial/DGPS modalities. | `src/features/provenance.py`, `configs/feature_provenance.yaml`, `src/risk/xgboost_model.py`, `src/anomaly/isolation_forest.py` | Registry completeness for Groups A–J; XGBoost/IF candidate exclusion tests; active feature-schema drift guard | No provenance boundary → only observable/config-derived, ungated columns are allowed; C/G/H/I remain gated |
| P2-ORACLE | Feature-store default was oracle and the production pipeline explicitly requested it; G-5 was incorrectly marked resolved in the prior audit. | `src/features/build_feature_store.py`, `src/features/group_c_spatial.py`, `src/pipeline.py`, `tests/test_group_c.py`, `tests/test_build_feature_store.py` | Oracle mode raises; parquet round-trip records detected/gated mode | Default/explicit oracle mode could flow to model features → default detected; all oracle model builds refused; G-5 reopened and corrected |

## Not fixed and why

- §21.1 neighbor-confirmation behavior is not empirically validated: the source corpus has one node per event. The pipeline fails closed with zero confirmations; a multi-node corpus is needed to evaluate the spatial condition.
- The old held-out test is burned. A new locked test corpus is required before any final evaluation; none has been read.
- Fixed physics priors are config-derived and currently match the synthetic generator's global settings. Their deployment validity depends on site configuration being available and correctly maintained.

## New problems found

- The checkout contained an untracked `src/risk/artifacts.py` and subsequent registry/tabletop artifact changes appeared during this work. They are preserved and not yet audited or included in the Phase 0/1 commits.
- `reports/superseded_leaky/` preserves v1 model outputs only for traceability; all are withdrawn as performance evidence. A clean tuning and final-evaluation run remains required.

## Tuning and finalization evidence

- `reports/tuning/baselines.json`: fixed three-fold event-grouped baseline comparison; logistic C selected from the light grid.
- `reports/tuning/isolation_forest_study.json`: 41/100 complete, 59 pruned; healthy-only fitting and validation-healthy percentile threshold.
- `reports/tuning/forecaster_study.json`: 59/60 complete, 1 pruned; next-window TCN beats grouped-CV persistence.
- `reports/tuning/xgboost_study.json`: 16 actual trials (12 complete, 4 pruned) after the user requested turnaround; 40 were requested and the 200+ plan was not reached. Independent CV revalidation passed the logistic false-alarm limit in all three folds.
- `reports/tuning/robustness.json`: shuffled-label chance control, learning curve, feature and IF ablations, SHAP, permutation, repeated 3-fold CV over three settings and three seeds, scenario recall, and workstation inference profile.
- `reports/pytest_phase7_hashseed_101.txt`, `_202.txt`, `_303.txt`: 625 tests passed at each seed. `scripts/check_tree.py` and `scripts/validate_synthetic_dataset.py` pass.
- Seeded tabletop stand-in is feature-scarce (18 missing features) and has Critical recall 0.0; it is not hardware or mine validation.

The fresh locked evaluation and its single-use lock update are recorded in `reports/final_eval.md` after freeze.
