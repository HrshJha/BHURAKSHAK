# Research and references slide copy

## Gap statement options

**A — cautious (39 words):** Despite periodic ground surveys and Sentinel-1 observations revisited every 6–12 days, a gap remains in continuous ground-level sensing and explainable risk analytics for coalfield subsidence; SubSense explores sensor fusion and risk scoring, while repository evidence remains synthetic and hardware remains a design.

**B — balanced (39 words):** Despite periodic surveys and Sentinel-1's 6–12-day revisit interval, a gap remains in continuous ground sensing paired with explainable subsidence-risk analytics; this project designs a sensor-to-alert workflow, with model evaluation limited to a locked synthetic scenario-regime holdout.

**C — strongest defensible (37 words):** Despite satellite revisit intervals of 6–12 days and periodic survey practice, a gap remains in continuous sensing plus timely, explainable risk scoring; our repository demonstrates the analytics on synthetic data, while the sensor network and gateway are still PRD designs.

Selected: **B**. Its revisit wording follows the repo's 12-day orbit repeat and 6-day interleaved-pair note; it does not imply a measured service gap or deployment.

## Six reference cards

Each card is tagged and ends in the cited DOI/product URL. Titles and takeaways are short enough for the specified card box.

1. **[GAP] Raniganj InSAR and DGPS** — *Surface deformation monitoring of Raniganj coalfield, India, using advanced InSAR and DGPS.* Ghosh et al.; Geomatics, Natural Hazards and Risk (2024). **Takeaway (17 words):** The Raniganj study supports regional InSAR context and DGPS ground control for our optional geodetic layer. [DOI](https://doi.org/10.1080/19475705.2024.2375546)
2. **[GAP] Korba PS-InSAR deformation** — *Identification and measurement of deformation using Sentinel data and PSInSAR technique in coalmines of Korba.* Monika et al.; ISPRS Archives (2018). **Takeaway (17 words):** Korba PS-InSAR provides a coalfield monitoring precedent; it does not validate our synthetic model or hardware. [DOI](https://doi.org/10.5194/isprs-archives-XLII-5-427-2018)
3. **[HW] SX1276 LoRa transceiver** — Semtech product page and datasheet. **Takeaway (17 words):** The SX1276 datasheet supports the PRD's selected LoRa radio; mesh routing and deployment remain designed. [Product page](https://www.semtech.com/products/wireless-rf/lora-connect/sx1276)
4. **[HW] MPU-9250 sensor datasheet** — InvenSense/TDK Product Specification. **Takeaway (16 words):** The MPU-9250 specification supports a PRD-listed IMU option; no physical sensor measurement is evidenced here. [Datasheet](https://invensense.tdk.com/wp-content/uploads/2015/02/PS-MPU-9250A-01-v1.1.pdf)
5. **[SW] Knothe subsidence time function** — Hejmanowski, “Modeling of time dependent subsidence for coal and ore deposits,” International Journal of Coal Science & Technology (2015). **Takeaway (18 words):** Knothe time dependence informs our configured synthetic deformation simulator; it is a modeling basis, not field validation. [DOI](https://doi.org/10.1007/s40789-015-0092-z)
6. **[SW] Isolation Forest and XGBoost** — Liu, Ting & Zhou (2008); Chen & Guestrin (2016). **Takeaway (17 words):** These papers ground the repository's anomaly and boosted-tree methods; our locked results are synthetic-regime evidence only. [Isolation Forest DOI](https://doi.org/10.1109/ICDM.2008.17) · [XGBoost DOI](https://doi.org/10.1145/2939672.2939785)

## Other documented info link box

- [SW] Public Data + ML repository and scope: [README](https://github.com/HrshJha/BHURAKSHAK/blob/54e6c82ee1580aaa3b7f76c50bc197dc213bc937/README.md)
- [SW] Reproduction commands: [reproduce guide](https://github.com/HrshJha/BHURAKSHAK/blob/54e6c82ee1580aaa3b7f76c50bc197dc213bc937/submit/reproduce.md)
- [SW] Notebook directory (no separate notebook index exists): [notebooks](https://github.com/HrshJha/BHURAKSHAK/tree/54e6c82ee1580aaa3b7f76c50bc197dc213bc937/notebooks)
- [SW] Synthetic dataset manifest and split: [manifest](https://github.com/HrshJha/BHURAKSHAK/blob/54e6c82ee1580aaa3b7f76c50bc197dc213bc937/data/synthetic/dataset_manifest.json)
- [SW] Research Document: [locked final evaluation](https://github.com/HrshJha/BHURAKSHAK/blob/54e6c82ee1580aaa3b7f76c50bc197dc213bc937/reports/final_eval.md)
- [HW] Hardware repo, BOM, photos, and demo link: none exists in this repository; request links from the hardware team.

## Figures and captions

- **fig: [HW] PRD-design node and gateway path.** Sensor suite → ESP32-WROOM-32 → SX1276/SX1278 LoRa → Raspberry Pi 5 gateway → uplink. All blocks are **DESIGNED (PRD only)**; this repository has no physical build evidence. Data origin: PRD design.
- **fig: [SW] Data + ML pipeline.** Packet validation → feature windows → Isolation Forest → spatial fusion → physics check → XGBoost → persistence alert engine → explainable output. Spatial neighbor confirmation is gated by single-node events. Data origin: synthetic corpus.
- **fig: [SW] End-to-end workflow.** Synthetic multi-sensor channels → preprocessing/features → anomaly, physics and risk layers → alert state → explainable operator output. No field integration is implied. Data origin: synthetic corpus.

## Locked synthetic results table

Unseen scenario-regime synthetic holdout (seed 20260929; one locked evaluation). Values are from the locked test only. Balanced accuracy is macro recall. `FA/day` is unavailable per classifier; the single tuned alert-engine aggregate is 0.046 false alarms per normal-event day and must not be compared across classifier rows.

| Model | Balanced acc. | CRITICAL precision | CRITICAL recall | Macro F1 | False alarms/day |
|---|---:|---:|---:|---:|---:|
| Threshold rule | 0.4692 | 0.1554 | 0.0416 | 0.3897 | n/a |
| Logistic regression | 0.4457 | 0.4904 | 0.7650 | 0.4297 | n/a |
| Default XGBoost | 0.5233 | 0.2469 | 0.0457 | 0.4450 | n/a |
| **Tuned XGBoost (ours)** | **0.4910** | **0.2632** | **0.0933** | **0.4458** | **0.046*** |

`*` Tuned alert-engine aggregate per normal-event day, not a per-model classifier rate. Negative median alert lead time: −3.33 h on the same synthetic holdout; not an early-warning success claim.

## Hardware spec and status table

Every row is **DESIGNED** from the PRD; no hardware is marked built or prototyped from this repository.

| Component | Part / interface | Key spec / source | Status | Evidence |
|---|---|---|---|---|
| Tilt / vibration IMU | MPU-6050 or MPU-9250; I²C | PRD options; MPU-9250 datasheet linked above | DESIGNED | `PRD.md` §8 |
| Displacement | VL53L1X ToF; I²C | PRD option; no measured value claimed | DESIGNED | `PRD.md` §8 |
| Node compute | ESP32-WROOM-32 | PRD selection | DESIGNED | `PRD.md` §8 |
| Node radio | SX1276/SX1278 LoRa; SPI | PRD selection; no link test evidenced | DESIGNED | `PRD.md` §8 |
| Gateway | Raspberry Pi 5 + LoRa HAT | PRD design; no gateway run evidenced | DESIGNED | `PRD.md` §26 |
| Power | Battery supply | PRD design; battery/runtime unspecified here | DESIGNED | `PRD.md` §8 |

**Chart omitted:** the “tabletop” dataset is a generated synthetic stand-in, not a physical rig log; model-specific false alarms/day are not retained. Neither requested chart can be supported by this repository.


## Copy length check

- Gap option A: 42 words.
- Gap option B: 37 words.
- Gap option C: 40 words.
- Card titles: all at most 8 words; takeaways: 16–18 words each. URLs and citation metadata are excluded from the word count.
