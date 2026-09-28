# Ablation

| removed_group | macro_pr_auc | critical_recall | macro_f1 | split | source_file | source_key |
| --- | --- | --- | --- | --- | --- | --- |
| All active inputs | 0.9312174331 | 0.9301075269 | 0.7191591546 | development_validation | reports/tuning/robustness.json | learning_curve.full_train.validation_metrics |
| A_physical | 0.9319298004 | 1 | 0.603365387 | development_validation | reports/tuning/robustness.json | feature_group_ablations.A_physical.metrics |
| B_temporal | 0.913456096 | 1 | 0.5822779644 | development_validation | reports/tuning/robustness.json | feature_group_ablations.B_temporal.metrics |
| D_vibration | 0.9317587759 | 1 | 0.6178890003 | development_validation | reports/tuning/robustness.json | feature_group_ablations.D_vibration.metrics |
| E_sensor_health | 0.9215787011 | 1 | 0.5903847834 | development_validation | reports/tuning/robustness.json | feature_group_ablations.E_sensor_health.metrics |
| F_physics | 0.8623946821 | 0.9982078853 | 0.5039832556 | development_validation | reports/tuning/robustness.json | feature_group_ablations.F_physics.metrics |
| G_dgps | 0.9297280016 | 1 | 0.6139303241 | development_validation | reports/tuning/robustness.json | feature_group_ablations.G_dgps.metrics |
| I_terrain | 0.9307847739 | 1 | 0.6158924924 | development_validation | reports/tuning/robustness.json | feature_group_ablations.I_terrain.metrics |
