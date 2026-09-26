# SubSense Task Board — Data + ML Workstream

Generated from `prd.md` on 2026-09-13. Total tasks: **85**. Do not hand-edit statuses outside the agent — update via EXECUTE MODE only.

**Scope:** this board covers ONLY the Data + ML workstream. Hardware, LoRa networking, Raspberry Pi deployment, MQTT, FastAPI/backend, databases, dashboard/UI, cloud, OTA and alert hardware are excluded by workstream decision and are listed (not silently dropped) in the "Excluded by Scope" section at the end, together with genuine PRD gaps.

## Status legend
- `[ ]` pending
- `[~]` in-progress
- `[x]` done
- `[!]` blocked — see note

## Phase → §34 mapping
| Phase | Content | Maps to §34 |
|---|---|---|
| 0 | Scaffolding | pre-Month 1 |
| 1 | Synthetic-data gate (§10) — **hard prerequisite** | Month 1 (parallel ML track) |
| 2 | Schema, labels, data quality, feature store | Month 1–2 |
| 3 | Isolation Forest, XGBoost, risk/alert engine | Month 1 (models) → Month 3 |
| 4 | Sentinel-1, DGPS, terrain, validation splits, ablation | Month 4 |
| 5 | Forecasting, domain-gap validation, final pipeline | Month 5 |
| Future | Gated roadmap (§17, §20, §38) | post-MVP |

> **Reported deviation:** §34 places almost no ML content in Months 2–3 (those months are hardware/gateway/dashboard, all out of scope here). Data + ML work that §34 leaves implicit — feature store, physics engine, risk engine, validation harness — has been placed in Phases 2–3 rather than leaving our workstream idle. Phase ordering still respects every dependency the PRD states.

> **Hard dependency (PRD-mandated, §10 + §34 Month 1):** the three synthetic-data-gate deliverables **T-022, T-023, T-024** must be `[x] done` before ANY model-training task (T-043 Isolation Forest, T-046 XGBoost) starts. This is encoded explicitly on every affected task's `Depends on` line. Parallel-work speed does not override it.

---

## Phase 0 — Scaffolding

### [x] T-001 — Create Data+ML repository directory tree
- **PRD ref:** §33
- **Depends on:** none
- **Output:** `data/raw/{synthetic,sentinel1,dgps,sensors}/`, `data/processed/{synthetic,insar,dgps,sensors}/`, `data/features/`, `data/labels/`, `notebooks/`, `src/{simulator,preprocessing,features,anomaly,risk,forecasting,physics,geospatial,evaluation}/`, `models/{isolation_forest,xgboost,temporal_model}/`, `tests/`, `configs/`, `docs/`, `scripts/`, `scripts/check_tree.py`
- **Acceptance check:** `python scripts/check_tree.py` exits 0, asserting every §33 path in the Data+ML subset exists and that the out-of-scope paths `src/api/`, `dashboard/`, `deployment/raspberry_pi/` are absent.
- **Status:** done

### [x] T-002 — Pin Data+ML dependency set in requirements.txt
- **PRD ref:** §26 (edge software stack), §33
- **Depends on:** T-001
- **Output:** `requirements.txt`
- **Acceptance check:** `pip install -r requirements.txt` succeeds in a clean venv and `python -c "import numpy, scipy, pandas, sklearn, xgboost, torch, onnxruntime"` exits 0. FastAPI/MQTT/DB drivers are intentionally absent (out of scope).
- **Status:** done

### [x] T-003 — Author configs/sampling.yaml from §8.3 defaults
- **PRD ref:** §8.3, NFR-6
- **Depends on:** T-001
- **Output:** `configs/sampling.yaml`
- **Acceptance check:** `pytest tests/test_configs.py::test_sampling_matches_prd` passes, asserting tilt default 2 Hz (range 1–10), displacement 1 Hz, vibration on-node ≥100 Hz summarized to 1 Hz, crack 0.1 Hz, temperature 0.1 Hz, adaptive multiplier 3× — matching the §8.3 table exactly.
- **Status:** done

### [x] T-004 — Author configs/alerts.yaml from §21.1 escalation table
- **PRD ref:** §21.1, FR-7, NFR-6
- **Depends on:** T-001
- **Output:** `configs/alerts.yaml`
- **Acceptance check:** `pytest tests/test_configs.py::test_alerts_match_prd` passes, asserting the three transitions with thresholds 0.5 / 0.6 / 0.7, persistence 3 / 3 / 2 windows, de-escalation multiplier 1.5×, and CRITICAL's ≥2-neighbouring-node confirmation flag.
- **Status:** done

### [x] T-005 — Author configs/physics.yaml with the §10 parameter set
- **PRD ref:** §10, NFR-6, NFR-7
- **Depends on:** T-001
- **Output:** `configs/physics.yaml`
- **Acceptance check:** `pytest tests/test_configs.py::test_physics_params` passes, asserting all ten §10 parameters are present: `panel_center_x, panel_center_y, panel_width, panel_length, mine_depth, extraction_height, subsidence_factor, influence_radius, time_coefficient, maximum_subsidence`.
- **Status:** done

### [x] T-006 — Author configs/feature_schema_v1.yaml
- **PRD ref:** §11, §13, §10.1
- **Depends on:** T-001
- **Output:** `configs/feature_schema_v1.yaml`
- **Acceptance check:** `pytest tests/test_configs.py::test_feature_schema` passes, asserting every field listed in §11 and every feature group A–J of §13 is enumerated, and that `feature_schema_version == "v1"` (matching the §10.1 manifest field).
- **Status:** done

### [x] T-007 — Implement config loader with no-hardcoded-threshold guarantee
- **PRD ref:** NFR-6
- **Depends on:** T-003, T-004, T-005, T-006
- **Output:** `src/config.py`, `tests/test_config_loader.py`
- **Acceptance check:** `pytest tests/test_config_loader.py` passes; includes a repo-wide scan asserting none of the §21.1 threshold literals (`0.5`, `0.6`, `0.7`, `1.5`) appear as risk-threshold constants anywhere under `src/` outside the loader.
- **Status:** done

### [x] T-008 — Write README.md carrying scope and the Honesty Statement
- **PRD ref:** Honesty Statement (final section), §4
- **Depends on:** T-001
- **Output:** `README.md`
- **Acceptance check:** `grep -q "cannot yet predict the exact time a real mine would fail" README.md` exits 0, and the README lists the §4 non-goals verbatim.
- **Status:** done

### [x] T-009 — Stand up pytest harness
- **PRD ref:** NFR-4
- **Depends on:** T-001
- **Output:** `tests/__init__.py`, `pytest.ini` (or `pyproject.toml` test config)
- **Acceptance check:** `pytest tests/ -q` collects ≥1 test and exits 0.
- **Status:** done

---

## Phase 1 — Synthetic-Data Gate (§10) — HARD PREREQUISITE

### [x] T-010 — Implement spatial deformation kernel W(x,y)
- **PRD ref:** §10 (spatial model)
- **Depends on:** T-005, T-007, T-009
- **Output:** `src/simulator/spatial_model.py`, `tests/test_spatial_model.py`
- **Acceptance check:** `pytest tests/test_spatial_model.py` passes, asserting the implementation matches `exp(-[(x−x₀)²+(y−y₀)²]/2σ²)` to within 1e-9 at ≥20 sampled points and peaks at `(x₀,y₀)`.
- **Status:** done

### [x] T-011 — Implement Knothe-style temporal model W(t)
- **PRD ref:** §10 (temporal model)
- **Depends on:** T-005, T-007, T-009
- **Output:** `src/simulator/temporal_model.py`, `tests/test_temporal_model.py`
- **Acceptance check:** `pytest tests/test_temporal_model.py` passes, asserting `W(t) = W_max·(1 − e^(−c·t))` to within 1e-9, `W(0) == 0`, and `W(t)` monotonically approaches `W_max`.
- **Status:** done

### [x] T-012 — Compose the coupled deformation field W(x,y,t)
- **PRD ref:** §10 (core principle — physical coupling)
- **Depends on:** T-010, T-011
- **Output:** `src/simulator/deformation_field.py`, `tests/test_deformation_field.py`
- **Acceptance check:** `pytest tests/test_deformation_field.py` passes, asserting `W(x,y,t) == W(t)·spatial_kernel(x,y)` and that the field is driven entirely by `configs/physics.yaml` with no literal parameters in code.
- **Status:** done

### [x] T-013 — Derive tilt channels as spatial gradients of W
- **PRD ref:** §10 (derived channels — `tilt_x ≈ ∂W/∂x`, `tilt_y ≈ ∂W/∂y`)
- **Depends on:** T-012
- **Output:** `src/simulator/channels_tilt.py`, `tests/test_channels_tilt.py`
- **Acceptance check:** `pytest tests/test_channels_tilt.py` passes, asserting generated `tilt_x`/`tilt_y` match a finite-difference gradient of the field to within tolerance, and that tilt is NOT independently sampled (Pearson r with the analytic gradient > 0.99).
- **Status:** done

### [x] T-014 — Derive inter-node displacement and strain
- **PRD ref:** §10 (`strain ≈ Δd_ij / d_ij`)
- **Depends on:** T-012
- **Output:** `src/simulator/channels_strain.py`, `tests/test_channels_strain.py`
- **Acceptance check:** `pytest tests/test_channels_strain.py` passes, asserting strain is computed from the change in inter-node distance `d_ij → d'_ij` under the deformation field, and equals zero everywhere when `W_max == 0`.
- **Status:** done

### [x] T-015 — Implement the vibration channel as supporting-only evidence
- **PRD ref:** §10 (`vibration = normal_noise + vehicle/personnel disturbance + localized transient + high-frequency burst`)
- **Depends on:** T-012
- **Output:** `src/simulator/channels_vibration.py`, `tests/test_channels_vibration.py`
- **Acceptance check:** `pytest tests/test_channels_vibration.py` passes, asserting all four components are independently toggleable and that a vibration-only scenario yields label `NON_SUBSIDENCE`, never `SUBSIDENCE` (encoding "vibration alone ≠ subsidence").
- **Status:** done

### [x] T-016 — Implement sensor-fault injection (5 fault modes)
- **PRD ref:** §10 (injected sensor faults), §12 (`fault_label`)
- **Depends on:** T-013, T-014
- **Output:** `src/simulator/faults.py`, `tests/test_faults.py`
- **Acceptance check:** `pytest tests/test_faults.py` passes, asserting `BIAS`, `STUCK`, `DROPOUT`, `SPIKE`, `DRIFT` are each injectable and each tagged `SENSOR_FAULT` — satisfying the §35 requirement of ≥3 fault types.
- **Status:** done

### [x] T-017 — Implement packet-loss and communication-failure modes
- **PRD ref:** §10 (taxonomy: `DATA_QUALITY`, `COMMUNICATION_FAILURE`)
- **Depends on:** T-016
- **Output:** `src/simulator/comms_degradation.py`, `tests/test_comms_degradation.py`
- **Acceptance check:** `pytest tests/test_comms_degradation.py` passes, asserting packet loss produces `DATA_QUALITY`-labelled rows and total link loss produces `COMMUNICATION_FAILURE`, distinct from `SENSOR_FAULT`.
- **Status:** done

### [x] T-018 — Implement the 12-row scenario taxonomy engine
- **PRD ref:** §10 (event/scenario taxonomy table)
- **Depends on:** T-015, T-016, T-017
- **Output:** `src/simulator/scenarios.py`, `tests/test_scenarios.py`
- **Acceptance check:** `pytest tests/test_scenarios.py` passes, asserting all 12 §10 scenarios are generatable and map to exactly the labels in the PRD table (`NORMAL`, `SENSOR_FAULT`, `DATA_QUALITY`, `LOCAL_ANOMALY`, `NON_SUBSIDENCE`, `SUBSIDENCE`, `HIGH_RISK`, `CRITICAL`, `MIXED`, `COMMUNICATION_FAILURE`), including ≥5 deformation scenario classes per §35.
- **Status:** done

### [x] T-019 — Implement the 20×20 virtual-grid mesh generator
- **PRD ref:** §10 (scale: 20×20 grid / 400 virtual nodes)
- **Depends on:** T-018
- **Output:** `src/simulator/grid.py`, `tests/test_grid.py`
- **Acceptance check:** `pytest tests/test_grid.py` passes, asserting exactly 400 virtual nodes with local (x, y) coordinates are produced and grid dimensions are config-driven.
- **Status:** done

### [x] T-020 — Implement seeded reproducibility across the generator
- **PRD ref:** NFR-7, §10
- **Depends on:** T-019
- **Output:** `src/simulator/rng.py`, `tests/test_reproducibility.py`
- **Acceptance check:** `pytest tests/test_reproducibility.py` passes, asserting two generator runs with the same seed produce byte-identical output and different seeds do not.
- **Status:** done

### [x] T-021 — Implement dataset_manifest.json writer
- **PRD ref:** §10.1 (hard requirement), NFR-7, FR-14
- **Depends on:** T-020
- **Output:** `src/simulator/manifest.py`, `tests/test_manifest.py`
- **Acceptance check:** `pytest tests/test_manifest.py` passes, validating emitted manifests against the exact §10.1 field list (`dataset_version`, `generator_version`, `random_seed`, `physics_parameters`, `noise_parameters`, `fault_parameters`, `scenario_parameters`, `source_data_versions`, `feature_schema_version`, `split_definition`).
- **Status:** done

### [x] T-022 — **GATE 1/3** — Produce synthetic_nodes.csv
- **PRD ref:** §10 (deliverable 1 of 3)
- **Depends on:** T-021
- **Output:** `data/synthetic/synthetic_nodes.csv`, `data/synthetic/dataset_manifest.json`, `scripts/generate_synthetic_nodes.py`
- **Acceptance check:** `python scripts/generate_synthetic_nodes.py` produces the CSV with one row per node per timestep carrying the raw §11 channels, with ≥10,000 generated sequences per §10, and its row count matches the accompanying manifest.
- **Status:** done

### [x] T-023 — **GATE 2/3** — Produce synthetic_events.csv
- **PRD ref:** §10 (deliverable 2 of 3)
- **Depends on:** T-021
- **Output:** `data/synthetic/synthetic_events.csv`, `scripts/generate_synthetic_events.py`
- **Acceptance check:** `python scripts/generate_synthetic_events.py` produces the CSV with exactly the §10 event-metadata columns: `id`, `start_time`, `end_time`, `type`, `severity`, `center`, `max_deformation`, `rate`; every `id` referenced in `synthetic_nodes.csv` resolves.
- **Status:** done

### [x] T-024 — **GATE 3/3** — Prove physical coupling in notebook 01
- **PRD ref:** §10 (deliverable 3 of 3)
- **Depends on:** T-022, T-023
- **Output:** `notebooks/01_synthetic_data_generation.ipynb`
- **Acceptance check:** the notebook executes end-to-end without error and emits both a plot and a numeric statistic showing `tilt` tracks the spatial gradient of the deformation field (correlation > 0.95), demonstrating the channels are coupled rather than independently random.
- **Status:** done

---

## Phase 2 — Schema, Labels, Data Quality & Feature Store

### [x] T-025 — Implement the §11 dataset schema module
- **PRD ref:** §11
- **Depends on:** T-006, T-022
- **Output:** `src/preprocessing/schema.py`, `tests/test_schema.py`
- **Acceptance check:** `pytest tests/test_schema.py` passes, asserting a dataframe validator accepts exactly the §11 field list and rejects unknown/missing columns.
- **Status:** done

### [x] T-026 — Implement the §12 separated label schema
- **PRD ref:** §12
- **Depends on:** T-025
- **Output:** `src/preprocessing/labels.py`, `tests/test_labels.py`
- **Acceptance check:** `pytest tests/test_labels.py` passes, asserting `anomaly_label`, `fault_label`, `progression_label`, `risk_label` and the continuous targets are stored as separate fields and that no code path collapses them into a single binary flag.
- **Status:** done

### [x] T-027 — Build dataset quality + EDA notebook 02
- **PRD ref:** §33, §15 (class imbalance), §24
- **Depends on:** T-025, T-026
- **Output:** `notebooks/02_dataset_quality_and_EDA.ipynb`
- **Acceptance check:** the notebook executes end-to-end and reports per-class counts for every §10 scenario label, explicitly quantifying the imbalance toward `NORMAL` that §15/§24 cite as the reason accuracy is rejected as a metric.
- **Status:** done

### [x] T-028 — Implement packet-level data validation (FR-3)
- **PRD ref:** FR-3, §9
- **Depends on:** T-025
- **Output:** `src/preprocessing/validation.py`, `tests/test_validation.py`
- **Acceptance check:** `pytest tests/test_validation.py` passes, asserting missing, duplicate, out-of-order and corrupted records are each detected and flagged as first-class data-quality states before feature generation.
- **Status:** done

### [x] T-029 — Implement the §9.1 resampling and interpolation rule
- **PRD ref:** §9.1
- **Depends on:** T-028
- **Output:** `src/preprocessing/resample.py`, `tests/test_resample.py`
- **Acceptance check:** `pytest tests/test_resample.py` passes, asserting nearest-neighbour matching within ±0.5× the sampling interval, linear interpolation for single-sample gaps, and NO interpolation across gaps longer than 3 missed samples (those become `DATA_QUALITY` flags).
- **Status:** done

### [x] T-030 — Implement node clock-drift estimation and correction
- **PRD ref:** §9.1 (node clock drift handling)
- **Depends on:** T-029
- **Output:** `src/preprocessing/clock_drift.py`, `tests/test_clock_drift.py`
- **Acceptance check:** `pytest tests/test_clock_drift.py` passes, asserting per-node drift is estimated from node-stamp vs receipt-stamp, applied as a correction offset at feature-generation time, and written to a log rather than silently absorbed.
- **Status:** done

### [x] T-031 — Implement cross-modality temporal alignment tolerances
- **PRD ref:** §9.1 (cross-modality alignment), FR-13
- **Depends on:** T-029
- **Output:** `src/preprocessing/align_modalities.py`, `tests/test_align_modalities.py`
- **Acceptance check:** `pytest tests/test_align_modalities.py` passes, asserting InSAR aligns to the nearest feature window within ±12 h and DGPS within ±1 h, and that each aligned record retains its own observation timestamp so staleness is explicit.
- **Status:** done

### [x] T-032 — Implement the §9.2 dual coordinate reference system
- **PRD ref:** §9.2
- **Depends on:** T-019
- **Output:** `src/geospatial/crs.py`, `tests/test_crs.py`
- **Acceptance check:** `pytest tests/test_crs.py` passes, asserting every node carries both mesh-local projected (x, y) in metres and a WGS84 (lat, lon) pair, that a round-trip transform returns to within 0.1 m, and that transformation parameters (origin lat/lon, UTM zone) are stored per deployment.
- **Status:** done

### [x] T-033 — Implement sensor-health state detection
- **PRD ref:** §9 (drift, stuck sensors, sudden offsets), §8.4 (`DEGRADED` state)
- **Depends on:** T-028
- **Output:** `src/preprocessing/sensor_health.py`, `tests/test_sensor_health.py`
- **Acceptance check:** `pytest tests/test_sensor_health.py` passes, asserting drift, stuck-value and sudden-offset conditions each produce the correct health state (including `DEGRADED`) and that injected `SENSOR_FAULT` rows from T-016 are correctly separated from genuine ground movement.
- **Status:** done

### [x] T-034 — Implement the windowing engine (60 timesteps, stride 10)
- **PRD ref:** §10 (windowing for XGBoost)
- **Depends on:** T-029
- **Output:** `src/features/windowing.py`, `tests/test_windowing.py`
- **Acceptance check:** `pytest tests/test_windowing.py` passes, asserting window=60 / stride=10 (config-driven), that per-window statistics (mean, std, slope, velocity, acceleration) are emitted, and that a guard raises if raw per-timestep rows are passed toward a model input.
- **Status:** done

### [x] T-035 — Implement Feature Group A (Physical)
- **PRD ref:** §13 Group A, FR-4
- **Depends on:** T-034
- **Output:** `src/features/group_a_physical.py`, `tests/test_group_a.py`
- **Acceptance check:** `pytest tests/test_group_a.py` passes, asserting exactly `tilt_x, tilt_y, tilt_magnitude, displacement, strain` are produced with names matching §13.
- **Status:** done

### [x] T-036 — Implement Feature Group B (Temporal)
- **PRD ref:** §13 Group B, FR-4
- **Depends on:** T-034
- **Output:** `src/features/group_b_temporal.py`, `tests/test_group_b.py`
- **Acceptance check:** `pytest tests/test_group_b.py` passes, asserting rolling mean/std/min/max, slope, velocity, acceleration, trend, persistence and change-point score are produced, and that no rolling statistic crosses a train/test boundary.
- **Status:** done

### [x] T-037 — Implement Feature Group C (Spatial)
- **PRD ref:** §13 Group C, FR-4
- **Depends on:** T-034, T-032
- **Output:** `src/features/group_c_spatial.py`, `tests/test_group_c.py`
- **Acceptance check:** `pytest tests/test_group_c.py` passes, asserting `neighbor_mean, neighbor_std, neighbor_anomaly_fraction, spatial_coherence, local_gradient, local_strain, hotspot_density, distance_to_subsidence_center` are produced and that spatial coherence distinguishes a single-node disturbance from multi-node coherent movement.
- **Status:** done

### [x] T-038 — Implement Feature Group D (Vibration)
- **PRD ref:** §13 Group D, §8.3
- **Depends on:** T-034
- **Output:** `src/features/group_d_vibration.py`, `tests/test_group_d.py`
- **Acceptance check:** `pytest tests/test_group_d.py` passes, asserting RMS, peak, crest_factor, low/mid/high band energy and spectral_centroid are produced from on-node-summarised vibration (never raw high-rate samples, per §8.3).
- **Status:** done

### [x] T-039 — Implement Feature Group E (Sensor health)
- **PRD ref:** §13 Group E, FR-4
- **Depends on:** T-033
- **Output:** `src/features/group_e_health.py`, `tests/test_group_e.py`
- **Acceptance check:** `pytest tests/test_group_e.py` passes, asserting `battery, RSSI, SNR, packet_loss, missing_ratio, stuck_sensor_flag, drift_score` are produced with names matching §13.
- **Status:** done

### [x] T-040 — Implement the physics engine (physics_residual)
- **PRD ref:** §21, FR-15
- **Depends on:** T-012, T-034
- **Output:** `src/physics/consistency.py`, `tests/test_physics_consistency.py`
- **Acceptance check:** `pytest tests/test_physics_consistency.py` passes, asserting `physics_residual = observed_deformation − expected_deformation` where expected deformation reuses the same §10 influence-function/Knothe model, and that residual ≈ 0 when observation equals the model.
- **Status:** done

### [x] T-041 — Implement Feature Group F (Physics)
- **PRD ref:** §13 Group F, FR-15
- **Depends on:** T-040
- **Output:** `src/features/group_f_physics.py`, `tests/test_group_f.py`
- **Acceptance check:** `pytest tests/test_group_f.py` passes, asserting `expected_displacement, expected_tilt, physics_residual, physics_residual_velocity` are produced with names matching §13.
- **Status:** done

### [x] T-042 — Assemble the feature store and verify the 40–70 feature budget
- **PRD ref:** §13 (target ~40–70 engineered features), §33
- **Depends on:** T-035, T-036, T-037, T-038, T-039, T-041
- **Output:** `src/features/build_feature_store.py`, `notebooks/03_feature_engineering.ipynb`, `data/features/`
- **Acceptance check:** the notebook executes end-to-end and asserts the assembled first-iteration feature count falls within 40–70 inclusive, failing loudly if it does not.
- **Status:** done

---

## Phase 3 — Baseline Models, Risk Engine & Traceability

### [x] T-043 — Implement the Isolation Forest module
- **PRD ref:** §14, FR-5
- **Depends on:** T-022, T-023, T-024, T-042
- **Output:** `src/anomaly/isolation_forest.py`, `tests/test_isolation_forest.py`
- **Acceptance check:** `pytest tests/test_isolation_forest.py` passes, asserting the model trains ONLY on healthy-baseline rows, uses `n_estimators=300, contamination="auto", random_state=42` exactly per the §14 code block, and outputs `anomaly_score = -model.score_samples(X)`.
- **Status:** done

### [x] T-044 — Demonstrate anomaly-score separation in notebook 04
- **PRD ref:** §14, §35 (acceptance bullet 2)
- **Depends on:** T-043
- **Output:** `notebooks/04_isolation_forest.ipynb`
- **Acceptance check:** the notebook executes end-to-end and renders an anomaly-score distribution plot plus a numeric separation statistic showing injected anomalies are demonstrably separated from normal behaviour.
- **Status:** done

### [x] T-045 — Run the §14 Isolation Forest ablation (A/B/C)
- **PRD ref:** §14 (ablation plan)
- **Depends on:** T-043
- **Output:** `experiments/if_ablation_abc.json`, `reports/if_ablation_abc.md`
- **Acceptance check:** the run reports false-alarm rate for (A) physical-only, (B) +temporal, (C) +spatial and states explicitly whether adding spatial coherence reduced false alarms.
- **Status:** done

### [ ] T-046 — Implement the XGBoost risk classifier
- **PRD ref:** §15, FR-6
- **Depends on:** T-022, T-023, T-024, T-042, T-043
- **Output:** `src/risk/xgboost_model.py`, `tests/test_xgboost_model.py`
- **Acceptance check:** `pytest tests/test_xgboost_model.py` passes, asserting the model targets the 3-class MVP `risk_label`, consumes feature groups A–F plus the Isolation Forest `anomaly_score` and `physics_residual`, and emits per-class probabilities that sum to 1.
- **Status:** pending

### [ ] T-047 — Implement the §15 comparison baselines
- **PRD ref:** §15, §35 (acceptance bullet 3)
- **Depends on:** T-042
- **Output:** `src/risk/baselines.py`, `tests/test_baselines.py`
- **Acceptance check:** `pytest tests/test_baselines.py` passes, asserting threshold-rule, logistic-regression and random-forest baselines all train and predict on the same feature matrix as XGBoost.
- **Status:** pending

### [ ] T-048 — Implement probability calibration and calibration metrics
- **PRD ref:** §15 (calibrated probabilities), §24 (Brier, reliability curve, ECE)
- **Depends on:** T-046
- **Output:** `src/risk/calibration.py`, `tests/test_calibration.py`
- **Acceptance check:** `pytest tests/test_calibration.py` passes, asserting Brier score, a reliability curve and expected calibration error are computed, and that calibration is fitted on a validation split disjoint from training.
- **Status:** pending

### [ ] T-049 — Build the XGBoost risk-model notebook 05
- **PRD ref:** §15, §33, §35 (acceptance bullet 3)
- **Depends on:** T-046, T-047, T-048, T-068
- **Output:** `notebooks/05_xgboost_risk_model.ipynb`
- **Acceptance check:** the notebook executes end-to-end and demonstrates XGBoost beating threshold-rule and logistic-regression baselines on PR-AUC and F1 on the held-out unseen-parameter-regime split (never a random split).
- **Status:** pending

### [ ] T-050 — Implement the alert-engine state machine
- **PRD ref:** §21.1, FR-7
- **Depends on:** T-004, T-046
- **Output:** `src/risk/alert_engine.py`, `tests/test_alert_engine.py`
- **Acceptance check:** `pytest tests/test_alert_engine.py` passes, asserting the three §21.1 transitions fire only after their required consecutive-window persistence and that NO single reading can escalate to CRITICAL.
- **Status:** pending

### [ ] T-051 — Implement de-escalation hysteresis
- **PRD ref:** §21.1 (de-escalation), §35 (acceptance bullet 8)
- **Depends on:** T-050
- **Output:** `src/risk/hysteresis.py`, `tests/test_hysteresis.py`
- **Acceptance check:** `pytest tests/test_hysteresis.py` passes, asserting de-escalation requires 1.5× the escalation persistence window at each level and that a borderline oscillating input does not flap between states.
- **Status:** pending

### [ ] T-052 — Implement per-node → region risk roll-up
- **PRD ref:** §21.1 (spatial aggregation)
- **Depends on:** T-050
- **Output:** `src/risk/aggregation.py`, `tests/test_aggregation.py`
- **Acceptance check:** `pytest tests/test_aggregation.py` passes, asserting region state is the max over constituent node states, and that a single noisy node cannot produce a panel-wide CRITICAL without the ≥2-neighbour confirmation rule.
- **Status:** pending

### [ ] T-053 — Implement operator override logging
- **PRD ref:** §21.1 (manual override)
- **Depends on:** T-050
- **Output:** `src/risk/override_log.py`, `tests/test_override_log.py`
- **Acceptance check:** `pytest tests/test_override_log.py` passes, asserting an override records operator ID, reason and timestamp, and that the underlying model output is still recorded (never suppressed).
- **Status:** pending

### [ ] T-054 — Implement the explainability contribution breakdown
- **PRD ref:** §22, FR-15
- **Depends on:** T-046
- **Output:** `src/risk/explainability.py`, `tests/test_explainability.py`
- **Acceptance check:** `pytest tests/test_explainability.py` passes, asserting every risk output is accompanied by a contributing-signal breakdown (e.g. tilt velocity, neighbour anomaly count, displacement trend, InSAR agreement, physics residual) and that emitting a bare probability without a breakdown raises.
- **Status:** pending

### [ ] T-055 — Implement the model registry (FR-14)
- **PRD ref:** §30, FR-14, §10.1
- **Depends on:** T-021, T-043, T-046
- **Output:** `src/risk/model_registry.py`, `models/registry.json`, `tests/test_model_registry.py`
- **Acceptance check:** `pytest tests/test_model_registry.py` passes, asserting every logged prediction carries `model_name, model_version, feature_version, training_dataset_version, timestamp`, and that `training_dataset_version` resolves to an existing §10.1 manifest.
- **Status:** pending

---

## Phase 4 — External Modalities, Validation Splits & Ablation

### [ ] T-056 — Select and document the Sentinel-1 study region and orbit track
- **PRD ref:** §18 step 1
- **Depends on:** T-001
- **Output:** `docs/insar_study_region.md`, `configs/insar.yaml`
- **Acceptance check:** the document fixes one Jharia study region and ONE consistent acquisition geometry/orbit track, with the bounding box and track number recorded in `configs/insar.yaml`.
- **Status:** pending

### [ ] T-057 — Acquire the Sentinel-1 SLC scene stack
- **PRD ref:** §18 step 2
- **Depends on:** T-056
- **Output:** `data/raw/sentinel1/`, `scripts/download_sentinel1.py`
- **Acceptance check:** between 20 and 40 SLC scenes for the fixed track are downloaded from the Copernicus Data Space Ecosystem and an inventory manifest lists each scene's date, track and geometry.
- **Status:** pending

### [ ] T-058 — Implement the offline InSAR processing workflow
- **PRD ref:** §18 step 3 (never on the Raspberry Pi), §4 (non-goal)
- **Depends on:** T-057
- **Output:** `src/geospatial/insar_processing.py`, `docs/insar_workflow.md`
- **Acceptance check:** the workflow runs on the workstation producing interferograms from the SLC stack, and `docs/insar_workflow.md` records the toolchain, coherence masking, atmospheric-correction choice and reference-point selection so the run is reproducible.
- **Status:** pending

### [ ] T-059 — Derive the InSAR deformation time series
- **PRD ref:** §18 step 4
- **Depends on:** T-058
- **Output:** `data/processed/insar/deformation_timeseries.parquet`
- **Acceptance check:** the output contains LOS displacement, velocity and coherence per persistent-scatterer point over the full scene stack, with low-coherence pixels masked rather than silently included.
- **Status:** pending

### [ ] T-060 — Map satellite features onto the mesh node/grid representation
- **PRD ref:** §18 step 5, FR-13
- **Depends on:** T-059, T-032, T-031
- **Output:** `src/geospatial/insar_to_mesh.py`, `tests/test_insar_to_mesh.py`
- **Acceptance check:** `pytest tests/test_insar_to_mesh.py` passes, asserting InSAR values join the shared feature schema on the same node/grid spatial representation and each carries its own observation timestamp and staleness age.
- **Status:** pending

### [ ] T-061 — Implement Feature Group H (InSAR)
- **PRD ref:** §13 Group H, FR-13
- **Depends on:** T-060
- **Output:** `src/features/group_h_insar.py`, `tests/test_group_h.py`
- **Acceptance check:** `pytest tests/test_group_h.py` passes, asserting `LOS_displacement, LOS_velocity, LOS_acceleration, cumulative_displacement, coherence, spatial_gradient, local_hotspot_density` are produced with names matching §13.
- **Status:** pending

### [ ] T-062 — Build the Sentinel-1 integration notebook 08
- **PRD ref:** §18, §33
- **Depends on:** T-061
- **Output:** `notebooks/08_sentinel1_integration.ipynb`
- **Acceptance check:** the notebook executes end-to-end and renders the Jharia deformation map plus the joined mesh-aligned InSAR feature table.
- **Status:** pending

### [ ] T-063 — Implement the DGPS ingestion and residual pipeline
- **PRD ref:** §19, FR-13
- **Depends on:** T-031, T-032
- **Output:** `src/geospatial/dgps.py`, `tests/test_dgps.py`
- **Acceptance check:** `pytest tests/test_dgps.py` passes, asserting DGPS points ingest at sparse validation/control locations, produce a `mesh_vs_dgps_residual`, and are used as evaluation targets rather than primary at-scale training labels.
- **Status:** pending

### [ ] T-064 — Implement Feature Group G (DGPS/GNSS)
- **PRD ref:** §13 Group G
- **Depends on:** T-063
- **Output:** `src/features/group_g_dgps.py`, `tests/test_group_g.py`
- **Acceptance check:** `pytest tests/test_group_g.py` passes, asserting `vertical_displacement, horizontal_displacement, velocity, acceleration, mesh_vs_dgps_residual` are produced with names matching §13.
- **Status:** pending

### [ ] T-065 — Build the DGPS validation notebook 09
- **PRD ref:** §19, §33
- **Depends on:** T-064
- **Output:** `notebooks/09_dgps_validation.ipynb`
- **Acceptance check:** the notebook executes end-to-end and reports mesh-vs-DGPS displacement agreement and any detected systematic sensor bias.
- **Status:** pending

### [ ] T-066 — Implement Feature Group I (Terrain / mine geometry)
- **PRD ref:** §13 Group I
- **Depends on:** T-032
- **Output:** `src/features/group_i_terrain.py`, `tests/test_group_i.py`
- **Acceptance check:** `pytest tests/test_group_i.py` passes, asserting `elevation, slope, aspect, curvature, mine_depth, panel_distance, panel_geometry, overburden` are produced with names matching §13.
- **Status:** pending

### [ ] T-067 — Implement Feature Group J (Environmental) behind an ablation gate
- **PRD ref:** §13 Group J ("included **only if** an ablation shows predictive value"), §38
- **Depends on:** T-070
- **Output:** `src/features/group_j_environmental.py`, `tests/test_group_j.py`
- **Acceptance check:** `pytest tests/test_group_j.py` passes, asserting Group J features are excluded from the model feature set by default and can only be enabled by a config flag that records the ablation result justifying inclusion.
- **Status:** pending

### [ ] T-068 — Implement the §23 leakage-safe validation splits
- **PRD ref:** §23 (random row-shuffling explicitly disallowed)
- **Depends on:** T-042
- **Output:** `src/evaluation/splits.py`, `tests/test_splits.py`
- **Acceptance check:** `pytest tests/test_splits.py` passes, asserting all five §23 split types (time, spatial, node, event, synthetic-parameter-regime) are implemented, that no `node_id` or time block appears in both train and test, and that a random row shuffle raises.
- **Status:** pending

### [ ] T-069 — Implement the §24 metrics module
- **PRD ref:** §24
- **Depends on:** T-046
- **Output:** `src/evaluation/metrics.py`, `tests/test_metrics.py`
- **Acceptance check:** `pytest tests/test_metrics.py` passes, asserting precision/recall/F1/PR-AUC, MAE/RMSE/max-abs-error/bias, IoU and hotspot-localisation error, Brier/reliability/ECE, and false-alarms-per-day / median lead time / P10 lead time / missed-event rate are all computed, and that bare accuracy is not exposed as a headline metric.
- **Status:** pending

### [ ] T-070 — Run the mandatory §25 ablation study (A–F)
- **PRD ref:** §25 (mandatory), §17, §16
- **Depends on:** T-061, T-064, T-066, T-068, T-069
- **Output:** `experiments/ablation_a_to_f.json`, `reports/ablation.md`
- **Acceptance check:** the study reports, for each of A (sensors only), B (+spatial), C (+DGPS), D (+Sentinel-1), E (+DGPS+Sentinel-1), F (+physics), the effect on lead time, false alarms, deformation error, spatial accuracy and calibration — the evidence that gates whether G (temporal DL) and H (GNN) are ever built.
- **Status:** pending

### [ ] T-071 — Build the validation and ablation notebook 07
- **PRD ref:** §23, §24, §25, §33
- **Depends on:** T-070
- **Output:** `notebooks/07_validation_and_ablation.ipynb`
- **Acceptance check:** the notebook executes end-to-end and renders the full §24 metric table across every §23 split and every §25 ablation step completed to date.
- **Status:** pending

---

## Phase 5 — Forecasting, Domain-Gap Validation & Final Pipeline

### [ ] T-072 — Implement the tabletop ground-truth analysis harness
- **PRD ref:** §23.1
- **Depends on:** T-069
- **Output:** `src/evaluation/tabletop_protocol.py`, `docs/tabletop_ground_truth_protocol.md`, `tests/test_tabletop_protocol.py`
- **Acceptance check:** `pytest tests/test_tabletop_protocol.py` passes on recorded fixture data, asserting each `run_id` links rig actuator setting → independent reference measurement → raw sensor data → derived risk state, and that mesh-estimated vs reference displacement error is computed. *(Software harness only — operating the physical rig is out of workstream scope.)*
- **Status:** pending

### [ ] T-073 — Run domain-gap Stage 1 (synthetic → synthetic)
- **PRD ref:** §23.2 stage 1
- **Depends on:** T-068, T-069, T-046
- **Output:** `experiments/domain_gap_stage1.json`
- **Acceptance check:** the run reports model performance on the §23 held-out unseen-parameter-regime split, establishing internal validity within the simulator's parameter space.
- **Status:** pending

### [ ] T-074 — Run domain-gap Stage 2 (synthetic → tabletop)
- **PRD ref:** §23.2 stage 2, §35 (acceptance bullet 7)
- **Depends on:** T-072, T-073
- **Output:** `experiments/domain_gap_stage2.json`
- **Acceptance check:** the trained-on-synthetic model runs UNMODIFIED on real tabletop sensor data and its risk classification is scored against the §23.1 independent ground truth. *(Expected to block until real tabletop sensor data exists — hardware-dependent input.)*
- **Status:** pending

### [ ] T-075 — Publish the honest domain-gap report
- **PRD ref:** §23.2, §35 (acceptance bullet 7), Honesty Statement
- **Depends on:** T-074
- **Output:** `reports/domain_gap.md`
- **Acceptance check:** the report states Stage 2 performance side-by-side with the Stage 1 synthetic-split result and explicitly reports degradation if present — passing requires the degradation be stated, not omitted.
- **Status:** pending

### [ ] T-076 — Implement the temporal forecasting model
- **PRD ref:** §16 (Phase 2 — implemented AFTER Isolation Forest + XGBoost are validated)
- **Depends on:** T-049, T-071
- **Output:** `src/forecasting/temporal_model.py`, `tests/test_temporal_model_fc.py`
- **Acceptance check:** `pytest tests/test_temporal_model_fc.py` passes, asserting the model predicts a PHYSICAL quantity (future displacement / tilt / deformation velocity) and never "future danger" directly, with TCN/GRU/LSTM selectable and LSTM retained as the literature benchmark.
- **Status:** pending

### [ ] T-077 — Feed forecasts back through the XGBoost risk layer
- **PRD ref:** §16 (preserving the layered explainable design)
- **Depends on:** T-076
- **Output:** `src/forecasting/forecast_to_risk.py`, `tests/test_forecast_to_risk.py`
- **Acceptance check:** `pytest tests/test_forecast_to_risk.py` passes, asserting forecasted physical values enter the XGBoost risk layer as features and that the forecaster never emits a risk state directly.
- **Status:** pending

### [ ] T-078 — Make forecast horizons configurable
- **PRD ref:** §16 (next-window, 30 min, 1 hr, 6 hr, 24 hr), NFR-6
- **Depends on:** T-076
- **Output:** `configs/forecasting.yaml`, `tests/test_forecast_horizons.py`
- **Acceptance check:** `pytest tests/test_forecast_horizons.py` passes, asserting all five §16 horizons are selectable from config with none hardcoded.
- **Status:** pending

### [ ] T-079 — Build the temporal forecasting notebook 06
- **PRD ref:** §16, §33, §24
- **Depends on:** T-077, T-078
- **Output:** `notebooks/06_temporal_forecasting.ipynb`
- **Acceptance check:** the notebook executes end-to-end and reports MAE, RMSE, max absolute error and bias per horizon against a persistence baseline, stating whether the neural forecaster justifies its complexity.
- **Status:** pending

### [ ] T-080 — Profile inference latency and memory against the edge budget
- **PRD ref:** NFR-2, §33 (notebook 10)
- **Depends on:** T-043, T-046
- **Output:** `notebooks/10_inference_profiling.ipynb`, `reports/inference_profile.md`
- **Acceptance check:** the notebook measures p50/p95 inference latency and peak RSS for the Isolation Forest + XGBoost artifacts on a single feature window and records them against the documented edge budget. *(Portable profiling only — executing on actual Raspberry Pi 5 hardware is out of workstream scope; see Gap G-8 on the missing numeric budget.)*
- **Status:** pending

### [ ] T-081 — Assemble the final end-to-end Data+ML pipeline and verify §35
- **PRD ref:** §35, §7 (layered pipeline)
- **Depends on:** T-055, T-071, T-075
- **Output:** `src/pipeline.py`, `scripts/run_pipeline.py`, `reports/acceptance_criteria.md`
- **Acceptance check:** `python scripts/run_pipeline.py` runs the full chain (validation → features → Isolation Forest → spatial fusion → physics check → XGBoost → alert engine → explainability) on held-out data, and `reports/acceptance_criteria.md` marks each §35 criterion within Data+ML scope as met with a pointer to the artifact proving it.
- **Status:** pending

---

## Phase: future — Gated Roadmap (pending, blocks nothing)

### [ ] T-082 — GNN / spatiotemporal graph modelling
- **PRD ref:** §17, §25 step H, §38, §4 (non-goal for MVP)
- **Depends on:** T-070
- **Output:** `src/risk/gnn_model.py`
- **Acceptance check:** started ONLY if the §25 ablation shows engineered spatial features + XGBoost are insufficient; the gate decision must be recorded in `reports/ablation.md` before any code is written.
- **Status:** pending

### [ ] T-083 — NISAR L-band cross-validation layer
- **PRD ref:** §20, §38
- **Depends on:** T-061
- **Output:** `src/geospatial/nisar.py`
- **Acceptance check:** NISAR products ingest as a complementary non-real-time layer only, with the 36–72 h latency and provisional status recorded in the feature's staleness metadata; must never be wired as a real-time trigger.
- **Status:** pending

### [ ] T-084 — Expanded environmental feature group
- **PRD ref:** §38, §13 Group J
- **Depends on:** T-067
- **Output:** `reports/environmental_ablation.md`
- **Acceptance check:** rainfall / land-cover / LST features are adopted only if the ablation records a measured improvement in an operational metric.
- **Status:** pending

### [ ] T-085 — Real-mine calibration and geotechnical validation
- **PRD ref:** §38, §23.2 stage 3, Honesty Statement
- **Depends on:** T-075
- **Output:** `docs/real_mine_validation_plan.md`
- **Acceptance check:** the plan documents what real mine data and geotechnical expertise are required before any operational use, and explicitly states this stage is NOT attempted within the SIH prototype timeline.
- **Status:** pending

---

## Excluded by Scope (reported, not tasked)

These PRD requirements are real and remain part of the overall SubSense project, but fall outside the Data + ML workstream. They are listed here so nothing is silently dropped.

| PRD item | Reason excluded |
|---|---|
| FR-1 (transport half) | ESP32/LoRa telemetry ingestion is firmware/networking. *The data-side half — raw record schema and §8.3 sampling config — IS tasked (T-003, T-025).* |
| FR-2 | Node/link failure detection and multi-hop rerouting — LoRa networking |
| FR-8 | Offline operation and local buffering — gateway/edge software |
| FR-9 | OTA firmware update with verify-and-rollback |
| FR-10 | FastAPI REST endpoints (§28) |
| FR-11 (render half) | GIS dashboard rendering. *The explainability payload it displays IS tasked (T-054).* |
| FR-12 | Adaptive sampling — firmware/gateway control loop |
| NFR-1 | End-to-end sensor→dashboard p95 latency — system-level |
| NFR-2 (device half) | On-Pi execution. *Portable model profiling IS tasked (T-080).* |
| NFR-3 | Battery life / field deployment |
| NFR-5 | Node/gateway scalability architecture |
| NFR-8 | Communication-layer abstraction (LoRa↔Zigbee↔Wi-Fi) |
| §8, §8.1, §8.2 | Sensor hardware, BOM, ToF displacement mechanism |
| §8.4 | Physical calibration protocol. *Its output fields must exist in node metadata — folded into T-025.* |
| §9.3 | Data retention/storage sizing — database/ops |
| §26 | Raspberry Pi deployment |
| §27 | Offline-first gateway requirement |
| §28 | Backend/API |
| §29 | Database schema and tables |
| §31 | React/Next.js dashboard |
| §32 | Security, OTA, LoRa reliability |
| §33 `docker-compose.yml`, `src/api/`, `dashboard/`, `deployment/raspberry_pi/` | DevOps/backend/UI paths — deliberately NOT created (asserted absent in T-001) |
| §34 Months 2, 3, 5 hardware/dashboard/OTA deliverables | Other workstreams |
| §35 bullets 4, 5, 6 (tabletop demo, outage demo, OTA demo) | Require hardware and gateway. *Bullets 1, 2, 3, 7, 8 ARE tasked.* |

## Gaps — PRD requirements I could NOT fully turn into a task

Flagged rather than silently resolved, per the hard rules. **Several of these are contradictions that will block execution if not settled.**

- **G-1 (blocking, label vocabulary):** §10's scenario taxonomy emits labels `NORMAL / SENSOR_FAULT / DATA_QUALITY / LOCAL_ANOMALY / NON_SUBSIDENCE / SUBSIDENCE / HIGH_RISK / CRITICAL / MIXED / COMMUNICATION_FAILURE`, but §12's `risk_label` is `GREEN / WATCH / WARNING / CRITICAL`. **No mapping between the two vocabularies is specified.** T-018 and T-026 both need it.
- **G-2 (blocking, contradiction):** §14's code trains on `df[df["risk_label"] == "GREEN"]`, but §12 says the MVP starts with 3 classes `NORMAL / WARNING / CRITICAL` — in which `GREEN` does not exist. As written, the §14 filter returns an empty training set. T-043 is written against "healthy-baseline rows" pending resolution.
- **G-3 (blocking, contradiction):** §21.1's first transition tests `P(WATCH or higher) > 0.5`, but the 3-class MVP model (§15) has no `WATCH` class. The state machine cannot be evaluated against the MVP model without either a 4-class model or a redefined trigger.
- **G-4 (numeric consistency):** §10 specifies 10,000–50,000 generated sequences; §15 expects 100k–500k windows. Whether these reconcile depends on sequence length ÷ stride, which is not stated. T-022 targets the §10 figure.
- **G-5 (leakage risk):** §13 Group C includes `distance_to_subsidence_center`. For synthetic data the centre is known from config, but on real data the subsidence centre is the thing being detected — this feature risks being a target proxy, which §24-adjacent leakage discipline forbids. Needs an explicit rule for how it is computed at inference time.
- **G-6 (missing toolchain):** §18 step 3 requires reproducible InSAR processing but names no toolchain (SNAP / ISCE2 / MintPy), no coherence-mask threshold, no atmospheric-correction method and no reference-point rule. T-058 records these as decisions to be made, not as specified values.
- **G-7 (missing data source):** §19 assigns DGPS a central calibration/validation role but identifies no receiver, vendor, survey partner or data source. T-063/T-064 may block on absent input.
- **G-8 (missing numeric target):** NFR-2 says edge inference must run "within its compute/power budget" but gives no latency, memory or power number. T-080 can measure but has no threshold to pass/fail against.
- **G-9 (missing numeric targets):** §24 lists false-alarms/day, median lead time and P10 lead time as operational metrics, but §35 sets no target values for any of them, so "good enough" is undefined.
- **G-10 (missing source):** §13 Group I requires `mine_depth`, `panel_geometry` and `overburden`. These come from config for synthetic data, but no real-mine source (BCCL/CMPDI plans) is identified for the Jharia case.
- **G-11 (unreviewed reference):** §37's trailing note flags `tandfonline.com/doi/full/10.1080/19475705.2024.2375546` as "not yet reviewed for content" — it must be read and folded into §18/§37 during Phase 4, and is not yet reflected in any task.
- **G-12 (not in PRD):** the repo is not under version control, yet NFR-7 reproducibility and FR-14 traceability both assume versioned artifacts. `git init` is not a PRD requirement so no task was created — recommend raising it as a scope addition.
