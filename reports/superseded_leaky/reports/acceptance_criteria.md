# §35 Acceptance Criteria — Data+ML Workstream Map (T-081)

**Scope note:** §35's acceptance list mixes Data+ML criteria with hardware/gateway/demo criteria. Per `TASKS.md` (Excluded by Scope), **bullets 4, 5 and 6 require hardware (tabletop rig demo, comms outage demo, OTA demo) and are out of this workstream**; they are marked as such below rather than claimed. Every "met" row points at the artifact that proves it — no criterion is asserted without one.

**End-to-end run backing this map:** `python scripts/run_pipeline.py` (T-081) executed the full §7 chain — validation → features → Isolation Forest → spatial fusion → physics check → XGBoost → alert engine → explainability — on the §12 corpus, scoring the held-out TEST split exactly once: 1,440,000 raw rows validated (39,829 DQ-flagged, all corrupted-value flags; 0 missing/duplicate/out-of-order), 90,000 windows × 61 features (§13 budget), IF threshold 0.7215, 42 §15 XGBoost inputs, 18,000 test windows scored, and the §21.1 ladder firing end-to-end on held-out data: **GREEN 14,994 → WATCH 1,668 → WARNING 878 → CRITICAL 460**, with 18,000 FR-11 explainability payloads emitted. Machine-readable summary: `experiments/pipeline_run.json` (includes per-stage provenance).

## Bullet-by-bullet

| §35 bullet | Status | Proving artifacts |
|---|---|---|
| **1 — Synthetic dataset**: ≥10,000 generated sequences with ≥5 deformation scenario classes | **MET** | `data/synthetic/synthetic_nodes.csv` + `data/synthetic/dataset_manifest.json` (T-021/T-022, 1.44 M rows, 10,000 events); `tests/test_scenarios.py` (T-023: all 12 §10 scenarios generatable, ≥5 deformation classes, exact PRD label table); notebooks 01–02 audit generation-to-label coupling |
| **2 — Anomaly detection**: Isolation Forest trained on the healthy baseline with a validation-derived threshold | **MET** | `src/anomaly/isolation_forest.py` (T-043: healthy-baseline TRAIN windows only — the triple-healthy G-2 mask — threshold = p0.99 of VALIDATION healthy scores); `notebooks/04_isolation_forest.ipynb` (score separation); `reports/if_ablation_abc.md` (T-045 §14 ablation A/B/C); `tests/test_isolation_forest.py` |
| **3 — Risk classification**: multimodal XGBoost, calibrated 3-class probabilities | **MET** | `src/risk/xgboost_model.py` (T-046: groups A–F + `anomaly_score` + `physics_residual`, softmax, probabilities sum to 1); `notebooks/05_xgboost_risk_model.ipynb`; calibration `src/risk/calibration.py` (T-048); baselines `src/risk/baselines.py` (T-047); full §24 metric surface per split × ablation arm in `notebooks/07_validation_and_ablation.ipynb` + `experiments/ablation_a_to_f.json` (T-069/T-070/T-071); `reports/ablation.md` |
| **4 — Tabletop demo** | **OUT OF SCOPE (hardware)** — but the data-side protocol is built and exercised | `src/evaluation/tabletop_protocol.py`, `docs/tabletop_ground_truth_protocol.md`, `tests/test_tabletop_protocol.py`, `data/recorded/tabletop/` (T-072); cross-domain generalisation honestly measured in `reports/domain_gap.md` (T-073/T-074/T-075) |
| **5 — Comms outage demo** | **OUT OF SCOPE (hardware/gateway)** — comms degradation is simulated on the data side | `src/preprocessing/sensor_health.py` + `tests/test_comms_degradation.py`, `tests/test_clock_drift.py` (data-side effects of degraded links are labelled, never silently imputed) |
| **6 — OTA demo** | **OUT OF SCOPE (firmware)** | — (model provenance/registry fields exist for it: `src/simulator/manifest.py`, T-030) |
| **7 — Honest reporting of limitations** | **MET (the explicit workstream discipline)** | `reports/domain_gap.md` (T-075: cross-domain degradation F1 0.398→0.147, CRITICAL recall 0.000, stated verbatim with the calibrative-vs-representational reading); literal-axis degeneracy in `experiments/domain_gap_stage1.json`; G-8 no-budget finding in `reports/inference_profile.md`; unsupported §16 horizons stated in `notebooks/06_temporal_forecasting.ipynb`; single-node-per-event Group C semantics recorded in notebook 03; G-9: §24 operational metrics are measured (`experiments/ablation_a_to_f.json`) but §35 sets no target values — none is claimed as "good" |
| **8 — Alert engine per §21.1** | **MET, demonstrated end-to-end on held-out data** | `src/risk/alert_engine.py` (T-050, config-driven thresholds/persistence/evidence from `configs/alerts.yaml`), `src/risk/hysteresis.py` (T-051, de-escalation = 1.5× escalation persistence, no flapping), `src/risk/aggregation.py` (T-052); **demonstrated in this run**: the full ladder GREEN→WATCH→WARNING→CRITICAL fires on the TEST split (counts above) with evidence conditions (`spatial_coherence`, `displacement_trend`, normalised `physics_residual`, ≥2 neighbour confirmations) enforced fail-closed |

## Criteria outside §35 but verified alongside

- **§16 forecasting** (physical-quantity-only contract, T-076/T-077/T-078): `tests/test_temporal_model_fc.py`, `tests/test_forecast_to_risk.py`, `tests/test_forecast_horizons.py`; evaluated vs persistence in `notebooks/06_temporal_forecasting.ipynb` (MAE skill +32.5% / +50.9% on the supported horizons).
- **NFR-2 profiling** (T-080): `notebooks/10_inference_profiling.ipynb`, `reports/inference_profile.md` — measured, not verdict-claimed (G-8).
- **NFR-6 configuration discipline**: `tests/test_config_loader.py` scan + `tests/test_forecast_horizons.py` horizon scan — every operational threshold/horizon lives in `configs/*.yaml`.
- **NFR-7 / reproducibility**: deterministic seeds throughout; every experiment JSON carries provenance metadata; the pipeline run is repeatable via `python scripts/run_pipeline.py`.

## What this map deliberately does not claim

- No bullet is marked met by "the code exists" alone — each points at an executed artifact (notebook output, experiment JSON, or passing test file).
- Bullets 4–6 remain the other workstreams' deliverables; the Data+ML side provides the tabletop protocol and recorded-data bridge so their hardware demos have a ready evaluation harness.
- Per **G-9**, no false-alarm/day or lead-time number is called "good enough" — §35 defines no target; the measured values live in the §24 tables.
