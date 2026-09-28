# Isolation Forest feature ablation

Isolation Forest is fit on healthy training rows only. Each ablation threshold is the healthy validation score 99th percentile. Synthetic development results:

| Removed group | Anomaly PR-AUC | Recall at threshold | Healthy false-positive rate |
|---|---:|---:|---:|
| A_physical | 0.6220 | 0.3481 | nan |
| B_temporal | 0.5186 | 0.1785 | nan |
| C_spatial | no_active_features_to_drop | — | — |
| D_vibration | 0.6284 | 0.3617 | nan |
| E_sensor_health | 0.6129 | 0.3453 | nan |
| F_physics | 0.6278 | 0.3585 | nan |
| G_dgps | 0.6156 | 0.3383 | nan |
| H_insar | no_active_features_to_drop | — | — |
| I_terrain | 0.6110 | 0.3481 | nan |
| J_environmental | no_active_features_to_drop | — | — |

The feature list is restricted to observable/config-derived inputs. Spatial Group C stays gated because no co-temporal neighbors exist.
