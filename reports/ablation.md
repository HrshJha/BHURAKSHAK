# Leakage-controlled model ablations

These synthetic-corpus results use the v2 allow-listed feature set and development validation groups. They are not locked-test results.

## XGBoost feature groups

| Removed group | Macro PR-AUC | Critical recall | Macro F1 |
|---|---:|---:|---:|
| A_physical | 0.9319 | 1.0000 | 0.6034 |
| B_temporal | 0.9135 | 1.0000 | 0.5823 |
| C_spatial | no_active_features (gated or unavailable) | — | — |
| D_vibration | 0.9318 | 1.0000 | 0.6179 |
| E_sensor_health | 0.9216 | 1.0000 | 0.5904 |
| F_physics | 0.8624 | 0.9982 | 0.5040 |
| G_dgps | 0.9297 | 1.0000 | 0.6139 |
| H_insar | no_active_features (gated or unavailable) | — | — |
| I_terrain | 0.9308 | 1.0000 | 0.6159 |
| J_environmental | no_active_features (gated or unavailable) | — | — |

Removing Group F (config-derived physics) caused the largest macro PR-AUC decline. Group C, H, and J are inactive or gated on this one-node-per-event corpus.

Top TreeSHAP features:
- `physics_residual`: mean absolute SHAP 1.1155
- `physics_residual_velocity`: mean absolute SHAP 0.6957
- `rolling_min`: mean absolute SHAP 0.5179

| Removed top feature | Macro PR-AUC | Critical recall | Macro F1 |
|---|---:|---:|---:|
| physics_residual | 0.9117 | 1.0000 | 0.5928 |
| physics_residual_velocity | 0.9329 | 1.0000 | 0.6074 |
| rolling_min | 0.9294 | 1.0000 | 0.6127 |

Permutation importance, the learning curve, and repeated grouped-CV results are in `reports/tuning/robustness.json`.
