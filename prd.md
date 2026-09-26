# PRD: AI-Enabled Low-Cost Real-Time Mine Subsidence Monitoring, Prediction and Early Warning System for Underground Coal Mines in India

**Codename:** SubSense
**Team:** zero chill
**SIH 2026 Problem Statement:** 26025 (Ministry of Coal / Coal India Limited) — Category: Hardware — Theme: Smart Automation

---

## 1. Executive Summary

SubSense is a low-cost, distributed IoT + ML system that continuously monitors surface deformation above underground coal-mine panels, detects abnormal ground behavior, estimates a calibrated subsidence-risk state, and issues early warnings before deformation becomes hazardous.

The system is built around a distributed **ESP32 sensor mesh** (tilt, displacement, vibration, crack, temperature) communicating over a **self-healing LoRa multi-hop network** to a **Raspberry Pi edge gateway**. The gateway runs a layered ML pipeline — **Isolation Forest → XGBoost → (later) TCN/GRU forecasting** — fused with asynchronous geodetic context from **Sentinel-1 InSAR**, **DGPS/GNSS**, and (future) **NISAR**. Output is a four-level risk state (GREEN/WATCH/WARNING/CRITICAL) surfaced on a GIS dashboard with explainable alerts.

The primary AI claim, stated precisely to avoid overclaiming:

> The system detects abnormal, spatially coherent deformation and estimates a calibrated subsidence-risk state from multimodal sensor and geodetic observations. It does not claim exact time-to-collapse prediction.

Because no usable real-mine labeled dataset exists, the project's core technical strategy is to **build a physically-coupled synthetic sensor dataset first** (Steps 1–8 of the dataset roadmap), validate the pipeline on it, then progressively fuse in Sentinel-1/Jharia InSAR, DGPS, and eventually real ESP32 hardware data — replacing synthetic channels one at a time while keeping the feature schema fixed.

---

## 2. Problem Definition

Underground coal extraction removes support from overlying strata. Over time this can produce surface subsidence — sinking, tilting, and cracking of the ground above the worked panel — which threatens nearby communities, agriculture, forests, infrastructure, and mine operations.

Current practice in most Indian coal mines relies on **periodic manual surveys** rather than continuous monitoring, meaning:

- Detection is delayed relative to when deformation actually begins.
- Spatial coverage is limited to survey points, not the full panel.
- There is no automated correlation between sensor readings, satellite deformation data, and geodetic ground-truth.
- Warnings are reactive rather than continuously risk-driven.
- Sensor/monitoring infrastructure, where it exists, degrades over large deployments without remote maintenance.

SubSense targets these gaps directly with continuous, low-cost, spatially distributed sensing fused with AI-based risk estimation.

---

## 3. Goals

- Continuously sense tilt, displacement, vibration, and (optionally) crack width across a mine panel via a low-cost sensor mesh.
- Detect abnormal deformation patterns using unsupervised anomaly detection before committing to a risk classification.
- Classify multimodal risk state (GREEN/WATCH/WARNING/CRITICAL) using engineered features + anomaly score + (when available) satellite/geodetic context.
- Operate fully offline at the edge; sync to cloud opportunistically.
- Visualize live risk, deformation hotspots, and node health on a GIS dashboard.
- Generate a physically defensible synthetic dataset as the initial training substrate, with a clear path to real-data validation (Sentinel-1/Jharia, DGPS, ESP32 hardware).
- Remain buildable by a student team in ~5 months with commodity hardware.

## 4. Non-Goals

- Predicting the exact time or magnitude of a catastrophic collapse.
- Replacing certified geotechnical survey or regulatory subsidence assessment.
- Operating as a fully autonomous evacuation-triggering system — CRITICAL alerts recommend action; evacuation remains human-authorized.
- Building a production-scale multi-mine SaaS platform in the prototype phase (architecture should allow for it later, but MVP targets one panel).
- Raw SAR/InSAR processing on the Raspberry Pi (done externally/offline on a workstation).
- Mandatory GNN-based spatial modeling for MVP (reserved as future work, gated behind an ablation showing engineered spatial features are insufficient).

---

## 5. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-1 | System shall ingest tilt, displacement, vibration, crack, temperature, battery, and radio-link telemetry from each ESP32 node at a configurable base sampling rate. |
| FR-2 | System shall detect node/link failure and reroute data via multi-hop LoRa. |
| FR-3 | System shall validate incoming data (missing/duplicate/out-of-order/corrupted packets) before feature generation. |
| FR-4 | System shall compute a rolling feature set (temporal, spatial, sensor-health, physics-residual) per node per window. |
| FR-5 | System shall run Isolation Forest anomaly scoring on live feature vectors. |
| FR-6 | System shall run XGBoost risk classification producing a 4-class risk state and calibrated probabilities. |
| FR-7 | System shall expose a deterministic alert-engine state machine that requires persistence/confirmation before escalating risk level (no single-reading trigger to CRITICAL). |
| FR-8 | System shall operate and continue alerting with no internet connectivity, buffering data locally and syncing when connectivity returns. |
| FR-9 | System shall support OTA firmware updates from the Raspberry Pi to the ESP32 fleet, with verify-and-rollback. |
| FR-10 | System shall expose a REST API (FastAPI) for node status, latest readings, risk state, risk map, alerts, forecasts, and history. |
| FR-11 | System shall render a GIS dashboard (map, time series, alert panel, explainability panel) reflecting live risk state. |
| FR-12 | System shall support adaptive sampling — increasing sensing/transmission rate for nodes/regions flagged as elevated risk. |
| FR-13 | System shall support ingestion of asynchronous Sentinel-1 InSAR and DGPS features, mapped onto the same node/grid spatial representation, as optional context features for XGBoost. |
| FR-14 | System shall log every prediction with model name, version, feature-schema version, and training-dataset version for traceability. |
| FR-15 | System shall support a physics-consistency check (expected vs. observed deformation) as a feature and as an explainability signal. |

## 6. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | End-to-end sensor→dashboard latency (p95) target: sub-minute under normal load, degrading gracefully under LoRa congestion. |
| NFR-2 | Edge inference (Isolation Forest + XGBoost) must run on Raspberry Pi 5 within its compute/power budget — no cloud dependency for core safety loop. |
| NFR-3 | Sensor nodes must run on battery for extended field deployment; adaptive sampling exists specifically to conserve power in normal conditions. |
| NFR-4 | System must be modular: sensor firmware, network layer, gateway, ML, and dashboard must be independently developable/testable. |
| NFR-5 | System must be scalable from ~6–20 prototype nodes to a full-panel deployment by adding nodes/gateways without architecture changes. |
| NFR-6 | All risk thresholds, sampling rates, and alert escalation rules must be configuration-driven, not hard-coded. |
| NFR-7 | All synthetic datasets must store their generating simulation parameters for reproducibility (fixed random seeds). |
| NFR-8 | The communication layer must be abstracted so LoRa can be swapped for Zigbee/Wi-Fi mesh without touching upper layers. |

---

## 7. System Architecture

```text
                    SENTINEL-1 / NISAR
                          │
                       InSAR processing (offline/workstation)
                          │
                  Satellite Features
                          │
                          ▼
DGPS/GNSS ───────→ GEODETIC CONTEXT
                          │
                          │
ESP32 SENSOR MESH  ───────┤
   (tilt, disp,           │
   vibration, crack)      │
        │                 │
        ▼                 │
 Wireless Gateway (LoRa multi-hop, self-healing)
        │                 │
        ▼                 │
 Raspberry Pi Edge Layer ◄┘
        │
        ├── Data Validation & Sensor Health
        ├── Feature Engineering
        ├── Isolation Forest (anomaly score)
        ├── Spatial Coherence / Kriging
        ├── Physics Consistency Check
        ├── XGBoost Risk Model
        ├── TCN / GRU Forecasting (later phase)
        └── Alert Engine (state machine)
                 │
                 ▼
        GREEN / WATCH / WARNING / CRITICAL
                 │
                 ▼
       GIS Dashboard (React/Leaflet) + SMS/Email/App/Siren
```

**Design principle (explicit rejection of the naive approach):** the system is *not* `Sensors → LSTM → Alert`. It is a layered pipeline — `Sensors → Sensor QA → Feature Engineering → Anomaly Detection → Spatial Fusion → Physics Check → Risk Fusion → Forecasting → Risk State → Alert` — because a single end-to-end deep model gives no explainability, no graceful degradation when a modality is missing, and no way to distinguish sensor fault from real deformation. Satellite and DGPS data are asynchronous context/validation layers, never a replacement for the real-time mesh, because InSAR/DGPS have coarse temporal cadence unsuitable as the sole real-time trigger.

---

## 8. Sensor Architecture

**Node class:** ESP32-based, low-cost, battery-powered.

**Primary sensing channels per node:**
- IMU / accelerometer / inclinometer → tilt_x, tilt_y
- Vibration sensor (MEMS) → vibration_rms, vibration_peak
- Displacement/stretch measurement (inter-node distance change) → displacement, strain
- Optional crack-width sensor → crack_width
- Temperature

**Node metadata:** `node_id`, lat/lon or local x/y, timestamp, battery, RSSI, SNR, packet_loss, firmware_version, sensor_health_state.

**Communication:** LoRa/LoRaWAN preferred (long range, low power, works without cellular/Wi-Fi coverage). Zigbee/Wi-Fi mesh are documented fallbacks. **Decision → Reason → Alternative → Why rejected:** Decision: LoRa multi-hop mesh. Reason: mine sites are frequently outside reliable cellular coverage and need km-scale range at low power. Alternative considered: Wi-Fi mesh. Rejected because of range and power-consumption unsuitability for battery-powered field nodes. The communication layer is implemented behind an abstraction interface so this choice is not architecturally locked in.

**Prototype layout (from field-layout diagram):** a 6-node grid (N1–N6) placed around the surface projection of an underground working panel — 3 nodes on the near edge (N1, N2, N3) and 3 on the far edge (N4, N5, N6) — sized to bracket the panel boundary so deformation gradients across the panel are observable, not just at a single point. This 6-node arrangement is the physical prototype layout for demo/validation; the synthetic dataset generator targets a denser 100–400 virtual-node mesh for ML training (see §10).

### 8.1 Sensor Hardware Selection & Bill of Materials (BOM)

Candidate parts are named here so the "low-cost hardware" claim in the PS is backed by an actual costed spec, not a category list. Prices are rough MVP-sourcing estimates (India, per-unit, small-batch) and must be re-verified before freezing the BOM.

| Channel | Candidate part | Measurement range | Resolution | Default sampling rate | Interface | Est. cost (₹) |
|---|---|---|---|---|---|---|
| Tilt / inclination | MPU-6050 (6-axis IMU) or MPU-9250 (9-axis) | ±2g accel, ±250°/s gyro (configurable) | ~0.06° tilt (post-filter) | see §8.3 | I²C | 150–350 |
| Displacement (inter-node) | VL53L1X ToF laser rangefinder | up to 4 m | ±1–3 mm | see §8.3 | I²C | 600–900 |
| Vibration | SW-420 (threshold) for MVP1; ADXL345 accelerometer for MVP2 (amplitude/RMS-capable) | ADXL345: ±16g | ADXL345: 13-bit (~4 mg/LSB) | see §8.3 (high-rate burst, summarized) | I²C/SPI | 100–300 |
| Crack width (optional) | Linear potentiometer / draw-wire sensor across a crack-monitoring gauge | 0–50 mm typical | ~0.1 mm | 0.1 Hz | Analog (ADC) | 300–600 |
| Temperature | DS18B20 (digital, waterproof probe) | −55°C to +125°C | 0.0625°C | 0.1 Hz | 1-Wire | 80–150 |
| Compute | ESP32-WROOM-32 dev board | — | — | — | — | 350–500 |
| Radio | SX1276/SX1278 LoRa module (or ESP32+LoRa integrated board, e.g. Heltec/TTGO LoRa32) | up to a few km LoS at low power | — | — | SPI | 500–900 (or ~900–1200 for integrated LoRa32 board, which replaces the separate ESP32 line) |
| Power | 18650 Li-ion cell(s) + TP4056 charge module + solar trickle panel (optional) | — | — | — | — | 300–700 |
| Enclosure | IP65 weatherproof junction box | — | — | — | — | 200–400 |
| **Per-node total (approx.)** | | | | | | **~₹2,600–4,600** (crack sensor and solar panel optional; base tilt+displacement+vibration+temp+comms+power+enclosure node lands near the lower end) |

**Gateway BOM:** Raspberry Pi 5 (₹6,000–8,000) + LoRa concentrator/gateway HAT (₹1,500–3,000) + storage (microSD/SSD, ₹500–1,500) + enclosure/power (₹500–1,000) ≈ **₹8,500–13,500**.

**Prototype total (6 nodes + 1 gateway), rough order of magnitude:** ~₹24,000–40,000 — this is the number to defend to judges as "low-cost," not a per-unit component price in isolation. Exact figures must be confirmed against current vendor pricing before the BOM is frozen.

### 8.2 Displacement Measurement Mechanism (MVP Decision)

**Decision:** Time-of-Flight (ToF) laser ranging (VL53L1X-class) mounted between adjacent nodes, sighted along the ground surface, as the MVP inter-node displacement mechanism.

**Reason:** ToF is contactless (no mechanical linkage to break or drift under field conditions), has millimeter-class resolution at the 1–4 m node-spacing scale relevant to a tabletop/small-panel prototype, is available as a cheap I²C breakout, and integrates with ESP32 with minimal wiring.

**Alternatives considered and why rejected for MVP:**
- **Cable/string potentiometer** — accurate and simple, but requires a physical string run between nodes that is fragile in a field/demo environment and reads only 1-D extension, not a clean displacement vector.
- **Ultrasonic rangefinder (e.g., HC-SR04)** — cheaper than ToF but centimeter-class resolution is too coarse for the mm-to-cm deformation signals this system needs to detect early.
- **Optical/camera-based (e.g., marker tracking)** — higher potential accuracy but adds a compute/vision pipeline the ESP32 cannot run locally, and lighting-dependent reliability makes it unsuitable for an outdoor mine-panel MVP.
- **Rotary encoder on a taut wire** — similar fragility profile to the string potentiometer, plus added mechanical complexity.

ToF is therefore the MVP decision; the sensor abstraction layer (§8, communication note) should keep this swappable, since a real multi-hundred-meter mine deployment will eventually need InSAR/DGPS rather than inter-node ToF for displacement at that scale — ToF is explicitly a **prototype/tabletop-scale** mechanism, not the long-term field solution.

### 8.3 Sampling Rate Defaults (MVP configuration)

| Channel | Default rate | Rationale |
|---|---|---|
| Tilt (IMU) | 1–10 Hz (default 2 Hz) | Ground tilt changes slowly; higher end reserved for adaptive high-risk mode |
| Displacement (ToF) | 1 Hz | Matches expected deformation timescale; avoids saturating LoRa payload |
| Vibration | High-frequency local sampling (≥100 Hz) on-node, **never transmitted raw** — reduced on-node to RMS/peak/crest-factor and sent at 1 Hz | Keeps LoRa payload small while preserving the features actually used downstream (§13, Group D) |
| Crack width | 0.1 Hz (optional channel) | Crack propagation is slow relative to tilt/displacement |
| Temperature | 0.1 Hz | Used for drift compensation, not primary signal |
| Node metadata (battery/RSSI/SNR/packet_loss) | Attached to every uplink packet | No separate schedule — piggybacked on data packets |

All values above are **defaults, not fixed** — NFR-6 already requires they be configuration-driven; this table exists so the MVP has a defined starting configuration rather than an unspecified "configurable rate."

**Adaptive-sampling multiplier:** on escalation to WATCH or above (§21.1), tilt and displacement rates increase by a configurable multiplier (default 3×) for the affected node and its immediate spatial neighbors, then decay back to baseline after a configurable cooldown once risk de-escalates.

### 8.4 Calibration Methodology

Cheap MEMS/ToF sensors drift; the PRD requires an explicit calibration protocol rather than a bare mention of "calibration":

1. **Factory/bench calibration** — each IMU and ToF unit is characterized against a known-flat reference surface and a known reference distance before field/demo deployment; offsets are stored per `node_id` in the model registry (§30) alongside the sensor's calibration date.
2. **Zeroing** — on node boot (and on manual trigger from the dashboard), the current tilt/displacement reading is captured as the node's local zero-reference, since absolute ground angle is not the signal of interest — *change from baseline* is.
3. **Multi-angle tilt calibration** — the IMU is calibrated at a minimum of 3 known tilt angles (e.g., 0°, 15°, 30° on a calibration jig) to fit a linear (or low-order polynomial) correction curve, rather than trusting factory defaults alone.
4. **Displacement reference calibration** — the ToF unit is calibrated against a precision tape-measure or laser-reference distance at 2–3 known separations spanning the expected node-spacing range.
5. **Temperature compensation** — since the DS18B20 channel exists specifically to support this, tilt and ToF readings are corrected using a temperature-vs-drift curve captured during bench calibration (cheap MEMS sensors are known to drift with temperature).
6. **Periodic recalibration** — a configurable interval (default: every 30 days, or on-demand from the dashboard) triggers a re-zero; sensors whose recalibration offset exceeds a configurable threshold are flagged `sensor_health = DEGRADED` and surfaced on the node-health panel rather than silently trusted.

This calibration record (per-node offsets, calibration date, drift history) is part of what must be captured in the dataset/model traceability chain (§10.1, §30).

---

## 9. Data Architecture

Raw stream per node:
```text
timestamp, node_id, tilt_x, tilt_y, displacement, vibration, battery, RSSI, SNR
```

Required pipeline stages: `ingestion → validation → synchronization → missing-data handling → sensor health → feature generation → ML inference → storage`.

Must tolerate: missing packets, delayed packets, duplicated packets, corrupted data, out-of-order timestamps, sensor drift, stuck sensors, sudden sensor offsets — these are treated as first-class data-quality states, not edge cases patched in later, because in the synthetic-data phase (§10) they are deliberately injected as `SENSOR_FAULT` scenarios and must be distinguishable from real ground movement or Isolation Forest will misclassify sensor failure as ground failure.

### 9.1 Time Synchronization

- All stored and transmitted timestamps use **UTC**, converted to local display time only in the dashboard layer.
- The **Raspberry Pi gateway is the single time authority** for a deployment: on join, each ESP32 node syncs its clock from the gateway (NTP when the gateway has internet, otherwise the gateway's own RTC/boot time as a local reference); nodes do not trust their own free-running clock as ground truth over long periods.
- **Node clock drift handling:** each uplink packet carries the node's local timestamp; the gateway also stamps its own receipt time. The two are compared per node to estimate drift, and a drift-correction offset is applied during feature generation rather than trusting raw node timestamps indefinitely — this offset is logged, not silently absorbed.
- **Resampling/interpolation rule:** feature windows (§10, `window = 60 timesteps, stride = 10`) are built on a fixed grid (default: nearest-neighbor within ±0.5× the sampling interval, linear interpolation for single-sample gaps, no interpolation across gaps longer than 3 missed samples — those become `DATA_QUALITY` flags instead).
- **Cross-modality alignment:** Sentinel-1 InSAR passes and DGPS observations are asynchronous and low-frequency relative to the sensor mesh. They are aligned to the nearest sensor feature window by timestamp (default tolerance: InSAR ±12 hours, DGPS ±1 hour) and tagged with their own observation timestamp so staleness is visible rather than implied.

### 9.2 Spatial Coordinate Reference System

- **Mesh-local coordinates:** each deployment defines a local projected (x, y) coordinate system in meters, origin at a fixed reference node or panel corner, used for all inter-node distance/strain/gradient calculations (§8.2, §10) — projected coordinates avoid the distortion of doing meter-scale geometry directly in lat/lon.
- **Global coordinates:** every node also stores a WGS84 (lat, lon) pair for GIS-dashboard mapping and for alignment with Sentinel-1/DGPS/NISAR products, which are natively geographic.
- **Transformation:** a fixed local tangent-plane (e.g., UTM zone appropriate to the mine site, or a simple equirectangular approximation for small panel extents) converts between the two; the transformation parameters (origin lat/lon, UTM zone) are stored per deployment so synthetic and real datasets remain comparable.

### 9.3 Data Retention & Storage Estimate

Rough order-of-magnitude sizing for the 6–8 node prototype, to justify the SQLite (MVP) → PostgreSQL (scale) decision in §29:

```
Per-node raw payload (post-compression): ~60–100 bytes/reading
Readings/day/node (at §8.3 default rates, dominated by 1 Hz displacement + 2 Hz tilt): ~260,000
Raw storage/day/node: ~15–25 MB
8-node prototype: ~120–200 MB/day raw
Feature table (60-step windows, stride 10): ~1/10th the row count of raw → ~12–20 MB/day
```

At this rate, a 6-month prototype deployment is on the order of a few GB of raw data plus a few hundred MB of features — comfortably within SQLite/local-disk limits for the demo, which is why SQLite is acceptable for MVP (§29) while PostgreSQL/TimescaleDB is reserved for a scaled, multi-mine deployment where node count and retention period both grow by orders of magnitude. Retention policy default: raw readings retained 90 days locally then downsampled/archived; feature tables and risk predictions retained indefinitely (small footprint, high analytical value).

---

## 10. Synthetic-Data Generator (`src/simulator/`)

**Why synthetic-first is the correct strategy, not a shortcut:** there is no public dataset matching this sensor architecture, and the published literature on InSAR/mining-subsidence ML (LSTM deformation forecasting, XGBoost mining-subsidence prediction, SarNet's 10⁶-interferogram synthetic training set) itself relies on synthetic or simulator-generated deformation data because labeled real deformation events are scarce. This is the documented answer to a judge asking "why synthetic data?"

**Core principle — physical coupling.** All sensor channels must derive from one latent deformation field, never be generated independently:

```text
                 TRUE DEFORMATION FIELD  W(x,y,t)
        ┌─────────────┬────────┬────────────┐
        ↓             ↓        ↓            ↓
      tilt      displacement  strain     vibration (supporting only)
        └──────────── sensor observations ──────┘
                              ↓
                       noise / faults
                              ↓
                        ML DATASET
```

**Spatial model:** Gaussian/influence-kernel approximation over a virtual mining panel, consistent with probability-integral / influence-function concepts used in mining subsidence literature:

```
W(x,y,t) = W(t) · exp( -[(x−x₀)² + (y−y₀)²] / 2σ² )
```

**Temporal model (Knothe-style):**
```
W(t) = W_max · (1 − e^(−c·t))
```

Configurable parameters: `panel_center_x, panel_center_y, panel_width, panel_length, mine_depth, extraction_height, subsidence_factor, influence_radius, time_coefficient, maximum_subsidence`.

**Derived channels (never independently random):**
```
tilt_x   ≈ ∂W/∂x
tilt_y   ≈ ∂W/∂y
strain   ≈ Δd_ij / d_ij   (from inter-node distance change d_ij → d'_ij)
vibration = normal_noise + vehicle/personnel disturbance + localized transient + high-frequency burst
```
Vibration is explicitly treated as supporting evidence only: `vibration alone ≠ subsidence`, but `vibration + persistent deformation + neighboring movement` strengthens the risk signal.

**Injected sensor faults (become `SENSOR_FAULT` labels):** bias, stuck-sensor (flatlined readings), dropout (NaN runs), random spike, slow drift. Without these, Isolation Forest will treat any sensor malfunction as ground failure.

**Event/scenario taxonomy:**

| Scenario | Label |
|---|---|
| stable ground | NORMAL |
| slow drift / temperature drift | NORMAL |
| sensor bias / stuck / dropout / random spike / slow drift (sensor) | SENSOR_FAULT |
| packet loss | DATA_QUALITY |
| single-node disturbance | LOCAL_ANOMALY |
| vibration-only event | NON_SUBSIDENCE |
| slow subsidence | SUBSIDENCE |
| accelerating subsidence | HIGH_RISK |
| rapid subsidence | CRITICAL |
| irregular subsidence | SUBSIDENCE |
| multiple deformation zones | MIXED |
| communication failure | COMMUNICATION_FAILURE |

**Scale:** start with a **20×20 virtual grid (400 virtual nodes)**, 10,000–50,000 generated sequences — not millions. Generator must be designed so scale can grow later; every synthetic dataset stores its generation parameters and a fixed random seed for reproducibility.

**Windowing for XGBoost:** `window = 60 timesteps, stride = 10`, converted to per-window statistical/temporal features (mean, std, slope, velocity, acceleration) rather than fed as a raw sequence — raw `timestamp, node_id, tilt_x, tilt_y, ...` rows are never handed directly to XGBoost.

**Before model training begins**, the following three deliverables must be completed, in order (this gate is a hard prerequisite for §14/§15 model work, not a parallel-track suggestion — see the Month-1 plan note in §34):
1. `synthetic_nodes.csv` (raw per-node-per-timestamp channels)
2. `synthetic_events.csv` (event metadata: id, start/end time, type, severity, center, max deformation, rate)
3. A notebook proving the synthetic channels are physically coupled (e.g., tilt visibly tracks the spatial gradient of the deformation field, not random noise)

Once those three exist, Isolation Forest (§14) and XGBoost (§15) development can begin immediately.

### 10.1 Dataset Manifest (`dataset_manifest.json`)

Every generated synthetic dataset (and, later, every real-data snapshot) must ship with a manifest file so any dataset used for training is traceable and reproducible — this is what the model registry (§30) actually points to as `training_dataset_version`. Required fields:

```json
{
  "dataset_version": "v0.1.0",
  "generator_version": "sim-0.3.1",
  "random_seed": 42,
  "physics_parameters": {"panel_center_x": 0, "panel_center_y": 0, "panel_width": 0, "panel_length": 0, "mine_depth": 0, "extraction_height": 0, "subsidence_factor": 0, "influence_radius": 0, "time_coefficient": 0, "maximum_subsidence": 0},
  "noise_parameters": {"tilt_noise_std": 0, "displacement_noise_std": 0, "vibration_noise_model": ""},
  "fault_parameters": {"fault_types": ["BIAS", "STUCK", "DROPOUT", "SPIKE", "DRIFT"], "injection_rate": 0},
  "scenario_parameters": {"scenario_types": [], "counts_per_scenario": {}},
  "source_data_versions": {"sentinel1": null, "dgps": null, "hardware": null},
  "feature_schema_version": "v1",
  "split_definition": {"type": "synthetic_parameter_holdout", "train_range": {}, "test_range": {}}
}
```

This manifest is a **hard requirement**, not optional metadata — without it, the "unseen parameter regime" test split (§23) and the model-registry traceability requirement (FR-14, §30) cannot actually be verified after the fact.

---

## 11. Dataset Schema

```text
timestamp, node_id, x, y
tilt_x, tilt_y, tilt_magnitude
displacement, strain
tilt_velocity, tilt_acceleration
displacement_velocity, displacement_acceleration
vibration_rms, vibration_peak
neighbor_mean, neighbor_std, neighbor_anomaly_fraction, spatial_coherence
battery, RSSI, SNR, packet_loss
DGPS_displacement, DGPS_velocity
InSAR_displacement, InSAR_velocity, InSAR_coherence
physics_displacement, physics_residual
anomaly_score
progression_label, risk_label
```

Directory layout: `data/raw/`, `data/processed/`, `data/features/`, `data/labels/`, `data/synthetic/` — each further split by source (`synthetic/`, `sentinel1/`, `dgps/`, `sensors/`).

## 12. Label Schema

Labels are kept **separate**, never collapsed into one binary flag — collapsing loses the distinction between "sensor is broken" and "ground is moving," which is the single most important design decision in the labeling strategy.

```text
anomaly_label:    0 = normal / 1 = abnormal
fault_label:      NONE / BIAS / STUCK / DROPOUT / DRIFT
progression_label: STABLE / SLOW / ACCELERATING / RAPID
risk_label:       GREEN / WATCH / WARNING / CRITICAL   (start with 3 classes: NORMAL/WARNING/CRITICAL, expand later)
continuous targets: deformation, deformation_velocity, deformation_acceleration
```

Ground truth is always simulator-defined (for synthetic data) or DGPS/InSAR-defined (for real data) — never model-defined.

---

## 13. Feature Engineering (Feature Store)

| Group | Features |
|---|---|
| A — Physical | tilt_x, tilt_y, tilt_magnitude, displacement, strain |
| B — Temporal | rolling mean/std/min/max, slope, velocity, acceleration, trend, persistence, change-point score |
| C — Spatial | neighbor_mean, neighbor_std, neighbor_anomaly_fraction, spatial_coherence, local_gradient, local_strain, hotspot_density, distance_to_subsidence_center |
| D — Vibration | RMS, peak, crest_factor, low/mid/high band energy, spectral_centroid |
| E — Sensor health | battery, RSSI, SNR, packet_loss, missing_ratio, stuck_sensor_flag, drift_score |
| F — Physics | expected_displacement, expected_tilt, physics_residual, physics_residual_velocity |
| G — DGPS/GNSS | vertical_displacement, horizontal_displacement, velocity, acceleration, mesh_vs_dgps_residual |
| H — InSAR | LOS_displacement, LOS_velocity, LOS_acceleration, cumulative_displacement, coherence, spatial_gradient, local_hotspot_density |
| I — Terrain/mine geometry | elevation, slope, aspect, curvature, mine_depth, panel_distance, panel_geometry, overburden |
| J — Environmental (optional) | rainfall, temperature, land_surface_temperature, land_cover — included **only if an ablation shows predictive value** |

Target for the first model iteration: **~40–70 engineered features**, not hundreds — complexity is added only when justified by the ablation study (§25).

---

## 14. Isolation Forest Design

**Purpose:** answer "does this look abnormal?" — not "is this subsidence?" The two questions are kept separate throughout the architecture.

- Trained **only on data labeled NORMAL/GREEN** (unsupervised, healthy-baseline training).
- Input: physical + temporal + spatial + vibration + sensor-health feature groups.
- Output: `anomaly_score = -model.score_samples(X)`, fed forward as one input feature to XGBoost — not interpreted directly as subsidence probability.
- Ablation plan: (A) physical-only features, (B) + temporal, (C) + spatial — measured against whether adding spatial coherence reduces false alarms.

```python
X_normal = df[df["risk_label"] == "GREEN"][features]
model = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
model.fit(X_normal)
anomaly_score = -model.score_samples(X)
```

---

## 15. XGBoost Design

**Purpose:** multimodal risk classification — the primary first serious risk model.

- Target: `risk_label` (start with 3 classes NORMAL/WARNING/CRITICAL; expand to GREEN/WATCH/WARNING/CRITICAL once the 3-class model is validated).
- Inputs: engineered feature groups A–I (as available) + Isolation Forest anomaly score + physics residual.
- Output: calibrated class probabilities, e.g. `P(GREEN)=0.01, P(WATCH)=0.07, P(WARNING)=0.84, P(CRITICAL)=0.08`; the alert engine (§20), not the raw model output, decides the resulting action.
- First experiment target: ~50 features, 100k–500k windows, 3 classes, evaluated against threshold rules, logistic regression, and random forest baselines using PR-AUC/F1/recall/false-alarms-per-day (never accuracy alone, given severe class imbalance toward NORMAL).

---

## 16. Temporal Forecasting Design (Phase 2)

Implemented **after** Isolation Forest + XGBoost are validated on adequate sequence data — not in MVP.

- Candidates: TCN, GRU, LSTM (LSTM retained as benchmark because published mining/InSAR forecasting literature already uses it).
- Predicts a **physical quantity** (future displacement / tilt / deformation velocity), never "future danger" directly — forecasted values are then fed back through the XGBoost risk layer, preserving the same explainable layered design.
- Horizons configurable: next-window, 30 min, 1 hr, 6 hr, 24 hr.

---

## 17. Spatial / GNN Roadmap (Future Work — not MVP)

Model the sensor mesh as a graph (sensor = node, physical proximity = edge). Candidates: GCN, GraphSAGE, GAT, spatiotemporal GNN. **Explicitly gated:** only pursued if the ablation study (§25) shows engineered spatial features (neighbor_mean, spatial_coherence, etc.) + XGBoost are insufficient. Not required for MVP.

---

## 18. InSAR Pipeline (Sentinel-1)

1. Select one consistent study region (Jharia coalfield) and one consistent acquisition geometry/orbit track.
2. Download Sentinel-1 **SLC** data (retains amplitude+phase needed for interferometry) from the Copernicus Data Space Ecosystem — start with ~20–40 scenes, not hundreds.
3. Process InSAR externally (workstation/cloud) — **never on the Raspberry Pi**.
4. Generate deformation time series; extract LOS displacement, velocity, coherence.
5. Map satellite features onto the same node/grid spatial representation used by the sensor mesh, so they can join the shared feature schema.

Reference precedent: the Eastern Jharia Sentinel-1 study (Jan 2018–Jan 2021 SLC series, PS-point filtering, stable/subsiding/uplifting classification) is used as the methodology reference for this pipeline, not as a ready-made CSV. A second Jharia multimodal remote-sensing study is tracked as an additional reference. NASA/OPERA's Sentinel-1 DISP product (30 m posting) is noted as a convenient alternative dataset **but its validated coverage is currently North America only**, so it cannot be used for Jharia — the archive-and-process-yourself route remains the plan for this study area.

---

## 19. DGPS Pipeline

DGPS/GNSS receivers are deployed at a small number of **selected validation/control points**, functioning as high-confidence ground truth to:
- calibrate low-cost mesh nodes,
- validate vertical and horizontal displacement measured by the mesh,
- detect systematic sensor bias,
- produce evaluation targets for model validation (not primary training labels at scale, since DGPS point coverage is sparse relative to the mesh).

## 20. NISAR Integration

Treated as a **complementary, non-real-time research layer** — used for L-band deformation cross-validation and future expansion, not as a primary input. NISAR's public provisional L-band products became available July 20, 2026, with Level 1–3 products currently carrying 36–72 hour latency and remaining provisional pending further validation/reprocessing — this latency and provisional status is exactly why NISAR cannot be the real-time trigger and is scoped as future work.

---

## 21. Physics Engine

Computes `physics_residual = observed_deformation − expected_deformation`, where expected deformation comes from the same influence-function/Knothe-style model used in the synthetic generator (§10), now applied to live sensor data as a consistency check rather than a data source.

Interpretation logic (fed into XGBoost, not hard-coded as a rule):
- High anomaly score + **low** spatial/physics consistency → likely sensor fault, vibration disturbance, or communication artifact.
- High anomaly score + high spatial coherence + persistent displacement + physics agreement → increases risk classification confidence.

### 21.1 Alert Engine & State Machine

FR-7 requires that no single reading can escalate risk straight to CRITICAL. This section defines the concrete state machine so that requirement is implementable rather than aspirational.

```
GREEN ──(N_watch consecutive elevated windows)──► WATCH
WATCH ──(N_warn consecutive windows with rising spatial coherence + deformation)──► WARNING
WARNING ──(N_crit consecutive windows of persistent severe trend + multi-signal confirmation)──► CRITICAL
```

**Escalation defaults (all configuration-driven per NFR-6, these are MVP starting values):**

| Transition | Trigger condition | Persistence required |
|---|---|---|
| GREEN → WATCH | XGBoost `P(WATCH or higher) > 0.5` for a node/region | 3 consecutive windows |
| WATCH → WARNING | `P(WARNING or higher) > 0.6` **and** spatial_coherence above threshold **and** displacement trend positive | 3 consecutive windows |
| WARNING → CRITICAL | `P(CRITICAL) > 0.7` **and** physics_residual low (i.e., physically consistent) **and** confirmed by ≥2 neighboring nodes | 2 consecutive windows |

**De-escalation (hysteresis):** de-escalation requires a *longer* persistence window than escalation (default: 1.5× the escalation window count) at each level, so the system does not flap between states on borderline readings — recovery is deliberately slower than alarm.

**Spatial aggregation:** risk state is tracked per-node and rolled up to a region/panel-level state as the max over its constituent nodes' states, so one severely affected node cannot be diluted by averaging with quiet neighbors, but a single noisy node also cannot trigger a panel-wide CRITICAL without the multi-node confirmation rule above.

**Manual override:** an operator can force an acknowledgement/hold at a given level from the dashboard (e.g., during known maintenance/vibration events), logged with operator ID and reason, but cannot suppress the underlying model output from being recorded.

---

## 22. Multimodal Fusion & Alert Explainability

The dashboard's explainability panel must show *why* risk increased in terms of contributing signals, e.g.:
```
+ high tilt velocity
+ 5 neighboring nodes anomalous
+ increasing displacement
+ InSAR deformation agreement
+ physics residual low
```
The system must never present a bare probability ("AI = 92%") without this breakdown — this is a functional requirement (FR-11), not a UI nicety, because a black-box percentage is not actionable or defensible to a mine operator or judge.

---

## 23. Validation Strategy

Random row-shuffling across time is explicitly disallowed. Required splits:

| Split | Rule |
|---|---|
| Time split | early period → train, middle → validation, late → test |
| Spatial split | sector A/B → train, sector C → test |
| Node split | nodes 1–15 → train, nodes 16–20 → test |
| Event split | train on slow deformation, test on accelerating deformation |
| Synthetic split | hold out complete simulator parameter regimes/geometries (e.g., train σ=5–20, test σ=22–30) |

### 23.1 Ground-Truth Protocol for the Tabletop Experiment

The synthetic simulator has simulator-defined ground truth, and the eventual real-mine layer has DGPS/InSAR ground truth — but the **physical tabletop demo has neither**, and needs its own independent reference so the demo's claimed deformation is verifiable rather than asserted. MVP protocol:

- A precision **linear reference** (e.g., a dial indicator or a second, independently calibrated high-resolution displacement sensor not used by the mesh) is mounted on the tabletop rig at the induced-subsidence point to log actual physical sinkage/movement during each demo run.
- Before each experiment run, the tabletop rig's induced deformation is manually surveyed (ruler/caliper measurement of the mechanical actuator's displacement) and logged alongside the run's timestamp.
- Sensor-mesh output (displacement, tilt, risk state) for that run is compared against this independent reference — this comparison **is** the tabletop validation result, and should be reported (e.g., mesh-estimated vs. reference displacement, correlation/error) rather than only shown as a qualitative "the dashboard lit up" demo.
- Every tabletop run is logged with a `run_id` linking: rig actuator setting → reference measurement → raw sensor data → derived risk state, so any run can be reproduced and audited.

### 23.2 Domain-Gap Validation (Synthetic → Tabletop → Real Mine)

This is flagged explicitly because it is the project's most significant scientific weakness if left implicit: a model trained purely on synthetic data is not automatically valid on physical sensor data, and physical tabletop validation is not automatically valid on a real mine. The PRD therefore requires an explicit three-stage validation chain rather than assuming transfer:

1. **Synthetic → Synthetic (internal validity):** the §23 splits above, confirming the model generalizes within the simulator's own parameter space.
2. **Synthetic → Tabletop (sim-to-real, small scale):** the trained-on-synthetic model is run, unmodified, on real tabletop sensor data (§23.1) and its risk classification is compared against the tabletop's independent ground truth. A materially worse result here than on the synthetic test split is expected and must be reported, not hidden — it is the honest measure of the sim-to-real gap, and is exactly what the domain-adaptation step referenced in §10/§18 (feature-schema alignment with Sentinel-1) exists to eventually close.
3. **Tabletop → Real mine (future work, not MVP):** explicitly deferred, and explicitly gated on real mine data + geotechnical calibration per the Honesty Statement — this PRD does not claim Stage 3 is complete or attempted within the SIH prototype timeline.

Reporting Stage 2 results (even if they show degraded performance) is a required acceptance-criteria item (§35), because presenting only synthetic-split metrics without acknowledging the sim-to-real gap would overstate the system's current readiness.

---

## 24. Metrics

| Category | Metrics |
|---|---|
| Anomaly detection | Precision, Recall, F1, PR-AUC, false positives/day |
| Forecasting | MAE, RMSE, Max Absolute Error, Bias |
| Spatial | IoU, spatial precision/recall, hotspot localization error |
| Risk calibration | Brier score, reliability curve, expected calibration error |
| Operational | false alarms/day, median lead time, P10 lead time, missed-event rate, packet delivery ratio, p50/p95 latency, node uptime |

Accuracy alone is explicitly rejected as a reporting metric given severe class imbalance toward NORMAL.

## 25. Ablation Plan (Mandatory)

Evaluate incremental modality contribution:

```
A: sensors only
B: sensors + spatial features
C: sensors + DGPS
D: sensors + Sentinel-1
E: sensors + DGPS + Sentinel-1
F: sensors + DGPS + Sentinel-1 + physics
G: full system + temporal deep learning
H: full system + GNN
```

Report, for each step, the effect on lead time, false alarms, deformation error, spatial accuracy, and calibration — this is what determines whether GNN/deep temporal modeling is ever built (§17, §16), not intuition.

---

## 26. Raspberry Pi Deployment

**Hardware:** Raspberry Pi 5, as gateway, ingestion service, local DB, feature engine, anomaly + risk inference, local dashboard, alert service, offline buffer. **Training happens on workstation/cloud, not on-device.**

**Edge software stack:** Python, NumPy, SciPy, pandas, scikit-learn, XGBoost, PyTorch (for later temporal models), ONNX Runtime (inference), FastAPI, MQTT, SQLite/PostgreSQL. Optional: TimescaleDB, Redis, Docker.

**Edge data flow:**
```
ESP32 → LoRa → Raspberry Pi → MQTT → Python ingestion → feature engine
→ Isolation Forest → XGBoost → (optional TCN/GRU) → risk engine
→ local alert → cloud sync
```

## 27. Offline-First Requirement

Core safety loop (`collect → store → infer → alert`) must never depend on internet connectivity. When connectivity returns, historical data syncs to cloud. This is a hard architectural constraint, not a nice-to-have, given mine sites' unreliable connectivity.

---

## 28. Backend / API

FastAPI endpoints:
```
GET  /health
GET  /nodes
GET  /nodes/{id}
GET  /latest
GET  /risk
GET  /risk/map
GET  /alerts
GET  /forecast/{node_id}
GET  /history/{node_id}
POST /sensor-data
POST /sync
```
Each endpoint requires defined request/response schemas, an authentication placeholder, rate-limiting considerations, structured error handling, and a versioning strategy (e.g., `/v1/...`) so the API can evolve without breaking the dashboard or ingestion clients.

## 29. Database

PostgreSQL for scalable deployment; SQLite acceptable for MVP/local development. Tables: `nodes, sensor_readings, sensor_health, features, anomalies, forecasts, risk_predictions, alerts, events, dgps_observations, insar_observations, simulation_runs, model_versions`.

## 30. Model Registry

Every prediction traceable to `model_name, model_version, feature_version, training_dataset_version, timestamp` (FR-14). Artifacts stored under:
```
models/
├── isolation_forest/
├── xgboost/
└── temporal_model/
```

---

## 31. Dashboard (React / Next.js)

- **Main map:** sensor nodes, sensor health, deformation, risk zones, InSAR overlay layer.
- **Time series:** tilt, displacement, vibration, risk probability, forecast.
- **Alert panel:** GREEN/WATCH/WARNING/CRITICAL.
- **Explainability panel:** contributing-signal breakdown (§22) — never a bare probability.

## 32. Security / Reliability

- Authentication placeholder on all API endpoints (to be hardened post-MVP).
- LoRa multi-hop self-healing routing to tolerate individual node/link failure.
- OTA firmware updates with verify-and-rollback (never push unverified firmware fleet-wide).
- Local buffering with store-and-forward guarantees no data loss during connectivity outages.
- All risk thresholds configurable, versioned, and auditable — no silent threshold changes.

---

## 33. Repository Structure

```text
mine-subsidence-ml/
├── data/
│   ├── raw/{synthetic,sentinel1,dgps,sensors}/
│   ├── processed/{synthetic,insar,dgps,sensors}/
│   ├── features/
│   └── labels/
├── notebooks/
│   ├── 01_synthetic_data_generation.ipynb
│   ├── 02_dataset_quality_and_EDA.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_isolation_forest.ipynb
│   ├── 05_xgboost_risk_model.ipynb
│   ├── 06_temporal_forecasting.ipynb
│   ├── 07_validation_and_ablation.ipynb
│   ├── 08_sentinel1_integration.ipynb
│   ├── 09_dgps_validation.ipynb
│   └── 10_raspberry_pi_profiling.ipynb
├── src/
│   ├── simulator/
│   ├── preprocessing/
│   ├── features/
│   ├── anomaly/
│   ├── risk/
│   ├── forecasting/
│   ├── physics/
│   ├── geospatial/
│   ├── api/
│   └── evaluation/
├── models/
├── tests/
├── configs/
├── deployment/raspberry_pi/
├── docs/
├── dashboard/
├── scripts/
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 34. Development Phases (5-Month Plan)

| Month | Deliverable |
|---|---|
| 1 | One working ESP32 node (tilt, displacement, vibration, crack); point-to-point LoRa; bench sensor calibration (§8.4). **In parallel, on its own track:** complete the synthetic-data gate (§10 — generator + event metadata + physical-coupling proof), then begin Isolation Forest/XGBoost on synthetic data. The synthetic-data gate must finish before model training starts, but does not block hardware work, which proceeds independently. |
| 2 | Scale to 6–8 physical nodes (matching the N1–N6 prototype layout); multi-hop comms, node IDs, packet ack, RSSI/SNR monitoring; stand up Raspberry Pi gateway. |
| 3 | MQTT messaging, local DB, dashboard v1, offline buffering; build the physical tabletop subsidence demo model. |
| 4 | Controlled-experiment data collection on the tabletop model; anomaly detection + sensor fusion on real hardware data; spatial risk map; begin Sentinel-1/Jharia integration (Steps 9–10 of roadmap). |
| 5 | Adaptive sampling, OTA updates, full end-to-end integration/testing, ablation study, demo scenarios, documentation, SIH presentation prep. |

## 35. Acceptance Criteria (MVP / Demo)

- Physics-coupled synthetic dataset generated with ≥3 fault types and ≥5 deformation scenario classes, parameters stored for reproducibility.
- Isolation Forest trained on synthetic NORMAL data, demonstrably separating injected anomalies from normal behavior (visualized anomaly-score distribution).
- XGBoost 3-class model beating threshold-rule and logistic-regression baselines on PR-AUC/F1 on held-out synthetic test split (unseen parameter regime, not random split).
- Live tabletop demo: physical model induces controlled "subsidence," ≥6 sensor nodes report to dashboard, risk state visibly escalates with an explainability breakdown, and the alert engine requires persistence before reaching CRITICAL (not a single-reading trigger).
- System continues collecting/alerting with simulated internet outage, then syncs on reconnect.
- At least one OTA update successfully pushed and rolled back on request.
- Domain-gap validation (§23.2, Stage 2) reported: trained-on-synthetic model's performance on real tabletop sensor data, compared against the tabletop's independent ground truth (§23.1) — reported honestly, including if it shows degraded performance relative to the synthetic test split.
- Alert state machine (§21.1) demonstrated with de-escalation/hysteresis, not just escalation, in at least one demo run.

## 36. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| No real-mine labeled dataset available | Synthetic-first strategy (§10) with published-literature precedent; controlled tabletop physical model for real (if small-scale) deformation data |
| Sensor noise/calibration drift | Multi-sensor fusion, explicit fault injection/labeling in training data, location-specific dynamic baselines |
| LoRa link/node failure in field conditions | Self-healing multi-hop routing, packet-loss/RSSI/SNR monitoring, store-and-forward buffering |
| Battery life for long-term deployment | Adaptive sampling (normal vs. high-risk mode), power-budget-aware duty cycling |
| Prediction accuracy varies across geology | Explicitly scoped as a prototype limitation (§37); real deployment requires calibration with geotechnical expertise and real mine data before operational use |
| OPERA DISP-S1 not covering Jharia | Fall back to processing Sentinel-1 SLC archive directly for the study area, using Eastern Jharia study as methodology reference |
| NISAR latency/provisional status | Scoped strictly as complementary/future-work layer, never the real-time trigger |
| Overclaiming to judges ("we predict collapse") | Explicit non-goal (§4) and honesty statement (§37); system claim is limited to abnormal-deformation detection + calibrated risk state |
| Cheap MEMS/ToF sensors drift, undermining low-cost claim's reliability | Explicit calibration protocol (§8.4) with periodic recalibration and a `DEGRADED` sensor-health state, not a one-time factory-cal assumption |
| Sim-to-real gap invalidates synthetic-trained model on physical/real data | Explicit three-stage domain-gap validation (§23.2) with mandatory honest reporting of Stage 2 (synthetic→tabletop) results, not just synthetic-split metrics |

## 37. Research References

| # | Title | URL | Relevance |
|---|---|---|---|
| 1 | LSTM mine subsidence prediction | https://www.mdpi.com/2072-4292/15/11/2755 | LSTM applied to mining subsidence prediction |
| 2 | Improved LSTM InSAR deformation forecasting | https://www.nature.com/articles/s41598-024-83084-1 | Reference methodology for Phase 2 temporal forecasting |
| 3 | XGBoost mining subsidence (genetic-algorithm combined) | https://www.mdpi.com/2071-1050/14/16/10421 | Reference for XGBoost risk model design |
| 4 | 2025 SBAS-InSAR + STL-XGBoost mining subsidence | https://www.nature.com/articles/s41598-025-21818-5 | Most directly relevant precedent to SubSense's XGBoost+InSAR fusion design |
| 5 | Synthetic ML/InSAR deformation identification (SarNet) | https://doi.org/10.1029/2020GC009204 | Precedent for synthetic-data-first strategy (10⁶ synthetic interferograms) |
| 6 | EQ-INSAR (GitHub) | https://github.com/kcieslik/eq-insar | Synthetic InSAR tooling inspiration (earthquake, not mining — methodology only) |
| 7 | EQ-INSAR paper | https://doi.org/10.1016/j.softx.2026.102903 | Companion paper to #6 |
| 8 | Eastern Jharia Sentinel-1 deformation study | https://isprs-annals.copernicus.org/articles/X-4-2024/349/ | Primary Jharia InSAR methodology reference (Jan 2018–Jan 2021 SLC series) |
| 9 | Jharia multi-sensor remote-sensing research | https://doi.org/10.1016/j.jag.2021.102439 | Additional Jharia-area remote-sensing reference |
| 10 | Copernicus Sentinel-1 data collection | https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-1 | Sentinel-1 SLC download source |
| 11 | Sentinel-1 documentation | https://documentation.dataspace.copernicus.eu/Data/Sentinel1.html | Data-access reference |
| 12 | NISAR mission overview | https://science.nasa.gov/mission/nisar/ | Future-work L-band reference |
| 13 | NISAR data availability | https://hyp3-docs.asf.alaska.edu/nisar-docs/availability-overview/ | Latency/provisional-status reference (§20) |
| 14 | ISRO/BCCL/CMPDI Jharia MoU | https://www.isro.gov.in/NRSC_ISRO_MoU_BCCL_CMPDI.html | Institutional context for Jharia NISAR/Sentinel-1 program |
| 15 | NASA OPERA DISP-S1 product | https://data.nasa.gov/dataset/opera-surface-displacement-from-sentinel-1-static-layers-validated-product-version-1 | Alternative InSAR displacement product (North America only — not usable for Jharia, §18) |

*(Additional reference supplied outside the master-prompt list, relevant to InSAR/mining-subsidence methodology: https://www.tandfonline.com/doi/full/10.1080/19475705.2024.2375546 — not yet reviewed for content; recommend the team read and, if relevant, fold into §18/§37 during Phase 4 InSAR work.)*

## 38. Future Roadmap (Post-MVP)

- GNN-based spatiotemporal modeling of the sensor mesh, gated behind ablation results (§17, §25).
- Multi-mine, multi-gateway deployment with centralized sync (originally scoped as a non-goal for MVP, §4).
- Expanded NISAR L-band fusion once provisional-product validation matures.
- Formal calibration and validation with real mine data and geotechnical expertise, required before any operational (non-prototype) use — stated explicitly to avoid overclaiming readiness.
- TimescaleDB/Redis/Docker hardening for production-grade deployment.
- Expanded environmental feature group (rainfall, land cover, LST) if ablation justifies inclusion.

---

## Honesty Statement (carried through from the team's own framing — retained deliberately)

> This prototype can detect and analyze controlled ground movement on a lab-scale model. It cannot yet predict the exact time a real mine would fail. Before any real-mine use, risk thresholds and prediction accuracy must be tested and calibrated with real mine data and geotechnical expertise.

This statement is treated as binding for how every claim elsewhere in this PRD should be read — no section above should be interpreted as overriding it.
