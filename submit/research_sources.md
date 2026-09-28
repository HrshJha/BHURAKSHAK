# SubSense research papers and data sources

Links below point to the sources used for the research slide. The repository link uses a fixed commit so it remains tied to the reviewed source state. Publisher pages may restrict automated access; DOI links remain the canonical paper links.

## Data used by the ML work

| Source | Direct link | Why it is included |
|---|---|---|
| Synthetic dataset manifest | [dataset_manifest.json](https://github.com/HrshJha/BHURAKSHAK/blob/54e6c82ee1580aaa3b7f76c50bc197dc213bc937/data/synthetic/dataset_manifest.json) | Describes the generated corpus, seed, split, and data provenance. This is the ML workstream’s training/evaluation data basis; it is not an external public dataset. |
| Synthetic corpus generator | [generate_synthetic_nodes.py](https://github.com/HrshJha/BHURAKSHAK/blob/54e6c82ee1580aaa3b7f76c50bc197dc213bc937/scripts/generate_synthetic_nodes.py) | Source code for recreating the physics-based sensor corpus. The generated `synthetic_nodes.csv` is local/ignored and is not available as a public download at this commit. |
| Synthetic event records | [synthetic_events.csv](https://raw.githubusercontent.com/HrshJha/BHURAKSHAK/54e6c82ee1580aaa3b7f76c50bc197dc213bc937/data/synthetic/synthetic_events.csv) | Public event-level part of the generated corpus; use with the generator and manifest for context. |
| Tabletop fixture manifest | [tabletop dataset manifest](https://github.com/HrshJha/BHURAKSHAK/blob/54e6c82ee1580aaa3b7f76c50bc197dc213bc937/data/recorded/tabletop/dataset_manifest.json) | Identifies the tabletop fixture as generated synthetic stand-in data, not physical rig evidence. |
| Tabletop fixture windows | [processed_windowed_dataset.csv](https://raw.githubusercontent.com/HrshJha/BHURAKSHAK/54e6c82ee1580aaa3b7f76c50bc197dc213bc937/data/recorded/tabletop/processed_windowed_dataset.csv) | Used for a feature-scarce transfer/domain-gap check only; it is not the primary training corpus and does not represent a real rig run. |
| Locked evaluation report | [final_eval.md](https://github.com/HrshJha/BHURAKSHAK/blob/54e6c82ee1580aaa3b7f76c50bc197dc213bc937/reports/final_eval.md) | Reports the single locked test on the unseen synthetic scenario-regime holdout. Use this for the model comparison; do not substitute development or pre-fix metrics. |

**Data boundary:** the models were trained on physics-based synthetic sensor data. No public mine-sensor dataset or physical tabletop log is the training corpus. Sentinel-1 is optional context; this repository’s InSAR workflow uses simulated stacks, not a real Sentinel-1 stack as training data.

## Research papers and technical references

| Reference | Direct link | Why it is included |
|---|---|---|
| Ghosh et al., “Surface deformation monitoring of Raniganj coalfield, India, using advanced InSAR and DGPS” (2024) | [Publisher article](https://www.tandfonline.com/doi/abs/10.1080/19475705.2024.2375546) · [DOI](https://doi.org/10.1080/19475705.2024.2375546) | Regional InSAR and DGPS ground-control precedent for the project’s optional geodetic context. It is literature context, not our model’s training data or validation. |
| Monika et al., “Identification and measurement of deformation using Sentinel data and PSInSAR technique in coalmines of Korba” (2018) | [Full paper PDF](https://isprs-archives.copernicus.org/articles/XLII-5/427/2018/isprs-archives-XLII-5-427-2018.pdf) · [DOI](https://doi.org/10.5194/isprs-archives-XLII-5-427-2018) | Coalfield PS-InSAR precedent. The paper reports deformation up to about 30 mm/year; this is not a result from our system. |
| Hejmanowski, “Modeling of time dependent subsidence for coal and ore deposits” (2015) | [Springer article](https://link.springer.com/article/10.1007/s40789-015-0092-z) · [DOI](https://doi.org/10.1007/s40789-015-0092-z) | Describes the Knothe influence/time model that informs the configured synthetic deformation simulator. |
| Liu, Ting & Zhou, “Isolation Forest” (2008) | [IEEE paper / DOI](https://doi.org/10.1109/ICDM.2008.17) | Research basis for the repository’s Isolation Forest anomaly layer. |
| Chen & Guestrin, “XGBoost: A Scalable Tree Boosting System” (2016) | [ACM paper / DOI](https://doi.org/10.1145/2939672.2939785) · [arXiv preprint](https://arxiv.org/abs/1603.02754) | Research basis for the repository’s XGBoost risk classifier. The arXiv link is an accessible author preprint. |
| Semtech SX1276 LoRa transceiver documentation | [Official product page and datasheet links](https://www.semtech.com/products/wireless-rf/lora-connect/sx1276) | Checks the radio component named in the hardware PRD; it does not show that a LoRa mesh was built or tested. |
| InvenSense/TDK MPU-9250 product specification | [Official datasheet PDF](https://invensense.tdk.com/wp-content/uploads/2015/02/PS-MPU-9250A-01-v1.1.pdf) | Documents a PRD-listed IMU option; no physical MPU-9250 rig measurements are present in this workstream. |

## Related project evidence

- [README and workstream boundary](https://github.com/HrshJha/BHURAKSHAK/blob/54e6c82ee1580aaa3b7f76c50bc197dc213bc937/README.md)
- [InSAR workflow and simulated-stack limitation](https://github.com/HrshJha/BHURAKSHAK/blob/54e6c82ee1580aaa3b7f76c50bc197dc213bc937/docs/insar_workflow.md)
- [Tabletop domain-gap report](https://github.com/HrshJha/BHURAKSHAK/blob/54e6c82ee1580aaa3b7f76c50bc197dc213bc937/reports/domain_gap.md)
