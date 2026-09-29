# Feature leakage audit

The current scan covers the v2 synthetic feature store and compares it with the archived v1 store only to quantify the removed leak. The v1 metrics are invalid and are not model results.

- Active artifact: `data/features/features_v2.parquet` (90,000 windows; schema v2).
- Archived comparison: `reports/superseded_leaky/data/features/features_v1.parquet` (90,000 windows; `superseded_leaky`).
- Labels are present only as targets. `build_feature_store` passes a copy containing structural keys and sensor channels only to feature emitters.
- Group C has no finite values in v2: all 90,000 production snapshots have one node and no co-temporal neighbors. The schema gate excludes C from XGBoost and Isolation Forest.  neighbor confirmation is not validated by this corpus.
- Group B persistence and change-point values are causal prefixes; truncation tests prove future windows do not alter earlier outputs.
- Split exclusivity: 10,000 unique events and `5,012` generating-parameter groups; zero group overlap across train/validation/test.
- Neighbor confirmations use only nodes co-temporal within the same event/window; alert persistence is isolated by event.

## Flagged features and provenance review

AUC is anomaly ROC-AUC and the maximum one-vs-rest risk ROC-AUC over the three classes (allowing either score direction). Correlations use numeric anomaly labels and ordinal risk labels. Values are univariate screening signals, not proof of leakage.

| Feature | Provenance class | Sources | v1 |r| anomaly | v2 |r| anomaly | v1 |r| risk | v2 |r| risk | v1 anomaly AUC | v2 anomaly AUC | v1 max risk AUC | v2 max risk AUC | Review |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `tilt_x` | `observable` | `tilt_x` | 0.0019 | 0.0019 | 0.0019 | 0.0019 | 0.4877 | 0.4877 | 0.5325 | 0.5325 | no threshold flag |
| `tilt_y` | `observable` | `tilt_y` | 0.0052 | 0.0052 | 0.0065 | 0.0065 | 0.4952 | 0.4952 | 0.5108 | 0.5108 | no threshold flag |
| `tilt_magnitude` | `observable` | `tilt_magnitude` | 0.2443 | 0.2443 | 0.0129 | 0.0129 | 0.6702 | 0.6702 | 0.5132 | 0.5132 | no threshold flag |
| `displacement` | `observable` | `displacement` | 0.3623 | 0.3623 | 0.2771 | 0.2771 | 0.7974 | 0.7974 | 0.7647 | 0.7647 | no threshold flag |
| `strain` | `observable` | `strain` | 0.0003 | 0.0003 | 0.0020 | 0.0020 | 0.4741 | 0.4741 | 0.5250 | 0.5250 | no threshold flag |
| `tilt_x_rolling_mean` | `observable` | `tilt_x` | 0.0022 | 0.0022 | 0.0021 | 0.0021 | 0.4882 | 0.4882 | 0.5330 | 0.5330 | no threshold flag |
| `tilt_x_rolling_std` | `observable` | `tilt_x` | 0.2169 | 0.2169 | 0.0079 | 0.0079 | 0.6763 | 0.6763 | 0.5019 | 0.5019 | no threshold flag |
| `tilt_x_rolling_min` | `observable` | `tilt_x` | 0.0393 | 0.0393 | 0.0034 | 0.0034 | 0.4820 | 0.4820 | 0.5330 | 0.5330 | no threshold flag |
| `tilt_x_rolling_max` | `observable` | `tilt_x` | 0.0349 | 0.0349 | 0.0007 | 0.0007 | 0.4946 | 0.4946 | 0.5327 | 0.5327 | no threshold flag |
| `tilt_x_slope` | `observable` | `tilt_x` | 0.0054 | 0.0054 | 0.0015 | 0.0015 | 0.4896 | 0.4896 | 0.5279 | 0.5279 | no threshold flag |
| `tilt_x_velocity` | `observable` | `tilt_x, timestamp` | 0.0061 | 0.0061 | 0.0025 | 0.0025 | 0.4925 | 0.4925 | 0.5198 | 0.5198 | no threshold flag |
| `tilt_x_acceleration` | `observable` | `tilt_x, timestamp` | 0.0057 | 0.0057 | 0.0027 | 0.0027 | 0.5039 | 0.5039 | 0.5036 | 0.5036 | no threshold flag |
| `tilt_x_trend` | `observable` | `tilt_x, timestamp` | 0.0054 | 0.0054 | 0.0015 | 0.0015 | 0.4896 | 0.4896 | 0.5279 | 0.5279 | no threshold flag |
| `tilt_x_persistence` | `observable` | `tilt_x` | 0.1449 | 0.2238 | 0.0077 | 0.0049 | 0.6364 | 0.6357 | 0.5054 | 0.5053 | no threshold flag |
| `tilt_x_change_point_score` | `observable` | `tilt_x` | 0.0989 | 0.2020 | 0.0003 | 0.0006 | 0.5998 | 0.6633 | 0.5048 | 0.5021 | no threshold flag |
| `tilt_y_rolling_mean` | `observable` | `tilt_y` | 0.0051 | 0.0051 | 0.0068 | 0.0068 | 0.4962 | 0.4962 | 0.5130 | 0.5130 | no threshold flag |
| `tilt_y_rolling_std` | `observable` | `tilt_y` | 0.2197 | 0.2197 | 0.0176 | 0.0176 | 0.6777 | 0.6777 | 0.5196 | 0.5196 | no threshold flag |
| `tilt_y_rolling_min` | `observable` | `tilt_y` | 0.0428 | 0.0428 | 0.0097 | 0.0097 | 0.4892 | 0.4892 | 0.5133 | 0.5133 | no threshold flag |
| `tilt_y_rolling_max` | `observable` | `tilt_y` | 0.0323 | 0.0323 | 0.0037 | 0.0037 | 0.5027 | 0.5027 | 0.5125 | 0.5125 | no threshold flag |
| `tilt_y_slope` | `observable` | `tilt_y` | 0.0052 | 0.0052 | 0.0048 | 0.0048 | 0.4921 | 0.4921 | 0.5064 | 0.5064 | no threshold flag |
| `tilt_y_velocity` | `observable` | `tilt_y, timestamp` | 0.0040 | 0.0040 | 0.0002 | 0.0002 | 0.4961 | 0.4961 | 0.5055 | 0.5055 | no threshold flag |
| `tilt_y_acceleration` | `observable` | `tilt_y, timestamp` | 0.0006 | 0.0006 | 0.0011 | 0.0011 | 0.4998 | 0.4998 | 0.5013 | 0.5013 | no threshold flag |
| `tilt_y_trend` | `observable` | `tilt_y, timestamp` | 0.0052 | 0.0052 | 0.0048 | 0.0048 | 0.4921 | 0.4921 | 0.5064 | 0.5064 | no threshold flag |
| `tilt_y_persistence` | `observable` | `tilt_y` | 0.1466 | 0.2237 | 0.0041 | 0.0040 | 0.6387 | 0.6353 | 0.5203 | 0.5076 | no threshold flag |
| `tilt_y_change_point_score` | `observable` | `tilt_y` | 0.1002 | 0.2007 | 0.0088 | 0.0006 | 0.5964 | 0.6601 | 0.5134 | 0.5071 | no threshold flag |
| `rolling_mean` | `observable` | `displacement` | 0.3485 | 0.3485 | 0.2863 | 0.2863 | 0.7923 | 0.7923 | 0.7692 | 0.7692 | no threshold flag |
| `rolling_std` | `observable` | `displacement` | 0.3751 | 0.3751 | 0.2271 | 0.2271 | 0.7713 | 0.7713 | 0.6622 | 0.6622 | no threshold flag |
| `rolling_min` | `observable` | `displacement` | 0.3290 | 0.3290 | 0.2904 | 0.2904 | 0.7768 | 0.7768 | 0.7790 | 0.7790 | no threshold flag |
| `rolling_max` | `observable` | `displacement` | 0.3600 | 0.3600 | 0.2793 | 0.2793 | 0.7901 | 0.7901 | 0.7581 | 0.7581 | no threshold flag |
| `slope` | `observable` | `displacement` | 0.3631 | 0.3631 | 0.2219 | 0.2219 | 0.7180 | 0.7180 | 0.7318 | 0.7318 | no threshold flag |
| `velocity` | `observable` | `displacement, timestamp` | 0.3559 | 0.3559 | 0.2129 | 0.2129 | 0.7213 | 0.7213 | 0.7013 | 0.7013 | no threshold flag |
| `acceleration` | `observable` | `displacement, timestamp` | 0.1099 | 0.1099 | 0.0366 | 0.0366 | 0.4488 | 0.4488 | 0.5869 | 0.5869 | no threshold flag |
| `trend` | `observable` | `displacement, timestamp` | 0.3631 | 0.3631 | 0.2219 | 0.2219 | 0.7180 | 0.7180 | 0.7318 | 0.7318 | no threshold flag |
| `persistence` | `observable` | `displacement` | 0.1400 | 0.3091 | 0.2477 | 0.1262 | 0.6843 | 0.6962 | 0.7438 | 0.5828 | no threshold flag |
| `change_point_score` | `observable` | `displacement` | 0.1144 | 0.2015 | 0.0682 | 0.0009 | 0.6348 | 0.7137 | 0.6929 | 0.5869 | no threshold flag |
| `neighbor_mean` | `observable` | `displacement, node_coordinates` | 0.3623 | n/a (gated/constant) | 0.2771 | n/a (gated/constant) | 0.7974 | n/a (gated/constant) | 0.7647 | n/a (gated/constant) | gated; no model input |
| `neighbor_std` | `observable` | `displacement, node_coordinates` | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | gated; no model input |
| `neighbor_anomaly_fraction` | `observable` | `displacement_slope, node_coordinates` | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | gated; no model input |
| `spatial_coherence` | `observable` | `displacement_slope, node_coordinates` | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | gated; no model input |
| `local_gradient` | `observable` | `displacement, node_coordinates` | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | gated; no model input |
| `local_strain` | `observable` | `displacement, node_coordinates` | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | gated; no model input |
| `hotspot_density` | `observable` | `displacement_slope, node_coordinates` | 0.9176 | n/a (gated/constant) | 0.1293 | n/a (gated/constant) | 0.9663 | n/a (gated/constant) | 0.5837 | n/a (gated/constant) | gated; no model input |
| `distance_to_subsidence_center` | `observable` | `displacement, node_coordinates` | 0.2314 | n/a (gated/constant) | 0.0092 | n/a (gated/constant) | 0.3385 | n/a (gated/constant) | 0.5126 | n/a (gated/constant) | gated; no model input |
| `vibration_rms` | `observable` | `vibration_rms` | 0.0065 | 0.0065 | 0.0098 | 0.0098 | 0.5050 | 0.5050 | 0.5086 | 0.5086 | no threshold flag |
| `vibration_peak` | `observable` | `vibration_peak` | 0.0045 | 0.0045 | 0.0048 | 0.0048 | 0.4952 | 0.4952 | 0.5043 | 0.5043 | no threshold flag |
| `crest_factor` | `observable` | `vibration_rms, vibration_peak` | 0.0006 | 0.0006 | 0.0067 | 0.0067 | 0.5004 | 0.5004 | 0.5054 | 0.5054 | no threshold flag |
| `band_energy_low` | `observable` | `vibration_rms` | 0.0066 | 0.0066 | 0.0026 | 0.0026 | 0.5048 | 0.5048 | 0.5035 | 0.5035 | no threshold flag |
| `band_energy_mid` | `observable` | `vibration_rms` | 0.0099 | 0.0099 | 0.0069 | 0.0069 | 0.5051 | 0.5051 | 0.5081 | 0.5081 | no threshold flag |
| `band_energy_high` | `observable` | `vibration_rms` | 0.0066 | 0.0066 | 0.0026 | 0.0026 | 0.4952 | 0.4952 | 0.5035 | 0.5035 | no threshold flag |
| `spectral_centroid` | `observable` | `vibration_rms` | 0.0066 | 0.0066 | 0.0026 | 0.0026 | 0.4952 | 0.4952 | 0.5035 | 0.5035 | no threshold flag |
| `battery` | `observable` | `battery` | 0.2115 | 0.2115 | 0.0000 | 0.0000 | 0.3521 | 0.3521 | 0.5006 | 0.5006 | no threshold flag |
| `RSSI` | `observable` | `RSSI` | 0.0110 | 0.0110 | 0.0037 | 0.0037 | 0.5077 | 0.5077 | 0.5043 | 0.5043 | no threshold flag |
| `SNR` | `observable` | `SNR` | 0.0049 | 0.0049 | 0.0059 | 0.0059 | 0.5037 | 0.5037 | 0.5098 | 0.5098 | no threshold flag |
| `packet_loss` | `observable` | `packet_loss` | 0.1007 | 0.1007 | 0.1201 | 0.1201 | 0.4493 | 0.4493 | 0.5581 | 0.5581 | no threshold flag |
| `missing_ratio` | `observable` | `sensor_channel_presence` | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | n/a (gated/constant) | no threshold flag |
| `stuck_sensor_flag` | `observable` | `displacement` | 0.0235 | 0.0235 | 0.1293 | 0.1293 | 0.4943 | 0.4943 | 0.5303 | 0.5303 | no threshold flag |
| `drift_score` | `observable` | `displacement` | 0.4788 | 0.4788 | 0.2350 | 0.2350 | 0.7096 | 0.7096 | 0.6854 | 0.6854 | no threshold flag |
| `expected_displacement` | `config_derived` | `node_coordinates, configs/physics.yaml` | 0.2011 | 0.2011 | 0.0108 | 0.0108 | 0.6720 | 0.6720 | 0.5126 | 0.5126 | no threshold flag |
| `expected_tilt` | `config_derived` | `node_coordinates, configs/physics.yaml` | 0.2487 | 0.2487 | 0.0127 | 0.0127 | 0.6747 | 0.6747 | 0.5128 | 0.5128 | no threshold flag |
| `physics_residual` | `config_derived` | `displacement, node_coordinates, configs/physics.yaml` | 0.1870 | 0.1870 | 0.2581 | 0.2581 | 0.6189 | 0.6189 | 0.7370 | 0.7370 | no threshold flag |
| `physics_residual_velocity` | `config_derived` | `displacement, timestamp, node_coordinates, configs/physics.yaml` | 0.2785 | 0.2785 | 0.1988 | 0.1988 | 0.6543 | 0.6543 | 0.7931 | 0.7931 | no threshold flag |

## Source audit findings

| Finding | Before | After | Evidence / disposition |
|---|---|---|---|
| `hotspot_density` read `anomaly_label` | v1 Pearson r = 0.9176 vs `anomaly_label`; exact same-window label-derived source | all v2 values NaN under the single-node gate | Removed label reads; uses a same-snapshot robust-z sensor-rate proxy only when co-temporal neighbors exist. Registry blocks it from current model inputs because the corpus has no neighbor graph. |
| `neighbor_anomaly_fraction` read `anomaly_label` | v1 was constant on one-node events, so correlation is undefined | all v2 values NaN under the gate | Same observable-only proxy when a real same-time graph exists; no cross-event neighbors. |
| Detected center read `anomaly_label`; default store requested oracle center | detected mode was label-dependent; oracle mode used simulator panel geometry | detected mode is recorded; center is based only on current observable displacement; oracle mode raises | `build_feature_store` rejects oracle mode and the pipeline requests detected mode. |
| Group B persistence/change-point used a whole-event scalar | same event-level value was copied to every window | recalculated from prefix through each window | Future-truncation tests compare earlier values byte-for-byte. |
| Pipeline confirmations grouped reused node IDs across events | unrelated events could confirm one another | counts restricted to same-event, same-window nodes; this corpus yields zero confirmations | `tests/test_pipeline_spatial.py` covers separate events with nearby coordinates and valid same-event neighbors. Alert state is keyed by event and node. |
| Group F physics expectation/residuals | configuration prior from `configs/physics.yaml`, not per-event sampled truth | same configured prior in v2 | Classified `config_derived`; deployment requires the same site configuration to be available. No per-event truth is read. |
| Group G DGPS residual and Groups G/H/I absent deployment feeds | not in the current first-iteration feature store | provenance gated from model inputs | DGPS agreement is evaluation-only; synthetic terrain and unavailable modalities stay gated. |

The script reads only the development feature store and the split metadata. It does not read `data/heldout_locked/` or any test predictions.
