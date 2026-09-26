# SubSense — AI-Enabled Mine Subsidence Monitoring, Prediction and Early Warning

**Team:** zero chill · **SIH 2026 PS 26025** (Ministry of Coal / Coal India Limited) · Category: Hardware · Theme: Smart Automation

SubSense is a low-cost, distributed IoT + ML system that continuously monitors surface deformation above underground coal-mine panels, detects abnormal ground behavior, estimates a calibrated subsidence-risk state (GREEN / WATCH / WARNING / CRITICAL), and issues early warnings before deformation becomes hazardous.

This repository contains the **Data + ML workstream**: the physics-coupled synthetic-data generator, feature store, Isolation Forest anomaly detection, XGBoost risk classification, physics-consistency engine, and the alert state machine. Hardware, LoRa networking, the Raspberry Pi gateway, MQTT, the FastAPI backend, databases, the GIS dashboard, OTA and alert hardware are separate workstreams (see `TASKS.md` → "Excluded by Scope").

> The primary AI claim, stated precisely to avoid overclaiming: the system detects abnormal, spatially coherent deformation and estimates a calibrated subsidence-risk state from multimodal sensor and geodetic observations. It does **not** claim exact time-to-collapse prediction.

## Honesty Statement

> This prototype can detect and analyze controlled ground movement on a lab-scale model. It cannot yet predict the exact time a real mine would fail. Before any real-mine use, risk thresholds and prediction accuracy must be tested and calibrated with real mine data and geotechnical expertise.

This statement is binding for how every claim in this repository should be read.

## Non-Goals (PRD §4, verbatim)

- Predicting the exact time or magnitude of a catastrophic collapse.
- Replacing certified geotechnical survey or regulatory subsidence assessment.
- Operating as a fully autonomous evacuation-triggering system — CRITICAL alerts recommend action; evacuation remains human-authorized.
- Building a production-scale multi-mine SaaS platform in the prototype phase (architecture should allow for it later, but MVP targets one panel).
- Raw SAR/InSAR processing on the Raspberry Pi (done externally/offline on a workstation).
- Mandatory GNN-based spatial modeling for MVP (reserved as future work, gated behind an ablation showing engineered spatial features are insufficient).

## Why synthetic data first

No usable real-mine labeled dataset exists, and the published InSAR/mining-subsidence ML literature itself relies on simulator-generated deformation data (see PRD §10). All sensor channels derive from one latent deformation field — Gaussian influence kernel over the panel, Knothe-style temporal growth — so tilt, displacement, strain and vibration are physically coupled, never independently random. The synthetic-data gate (three deliverables: `synthetic_nodes.csv`, `synthetic_events.csv`, and a physical-coupling proof in notebook 01) is a **hard prerequisite** for any model training.

## Setup

Requires Python 3.12 (pins in `requirements.txt` target ≥3.10; developed on 3.12.13).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
python3 scripts/check_tree.py
```

## Repository layout

```text
configs/       sampling.yaml, alerts.yaml, physics.yaml, feature_schema_v1.yaml (NFR-6: nothing hard-coded)
src/simulator/ physics-coupled synthetic-data generator (PRD §10)
src/preprocessing/  schema, labels, validation, resampling, clock drift (PRD §9, §11, §12)
src/features/  windowing + feature groups A–J (PRD §13)
src/anomaly/   Isolation Forest (PRD §14)
src/risk/      XGBoost, calibration, alert engine, explainability, registry (PRD §15, §21, §22, §30)
src/physics/   physics-consistency residual (PRD §21, FR-15)
src/geospatial/ CRS + InSAR/DGPS mapping (PRD §9.2, §18, §19)
src/evaluation/ leakage-safe splits + metrics (PRD §23, §24)
notebooks/     01–10 per PRD §33
data/          raw | processed | features | labels | synthetic
models/        isolation_forest | xgboost | temporal_model
```

## Governing documents

- `prd.md` — the PRD; the Honesty Statement is its final section and overrides optimistic readings elsewhere.
- `TASKS.md` — the 85-task Data+ML board with acceptance checks; statuses updated only via executed EXECUTE passes.
- `PROGRESS_LOG.md` — append-only execution log, one entry per pass.

All risk thresholds, sampling rates and escalation rules are configuration-driven (NFR-6) and asserted by test: `tests/test_config_loader.py` fails if §21.1 threshold literals appear under `src/` outside the loader.
