# Claims ledger

| claim_id | text_as_written | value | unit | source_file | key_or_line | script | data_origin | split | verified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C001 | SENSOR_FAULT: recall | 0.7949398443 | fraction | reports/tuning/robustness.json | required_scenario_category_recall.SENSOR_FAULT.recall_per_class.NORMAL | submit/build/make_submit.py | synthetic | development_validation | yes |
| C002 | SENSOR_FAULT: events | 628 | events | reports/tuning/robustness.json | required_scenario_category_recall.SENSOR_FAULT.events | submit/build/make_submit.py | synthetic | development_validation | yes |
| C003 | SENSOR_FAULT: recall | 0 | fraction | reports/tuning/robustness.json | required_scenario_category_recall.SENSOR_FAULT.recall_per_class.WARNING | submit/build/make_submit.py | synthetic | development_validation | yes |
| C004 | SENSOR_FAULT: recall | 0 | fraction | reports/tuning/robustness.json | required_scenario_category_recall.SENSOR_FAULT.recall_per_class.CRITICAL | submit/build/make_submit.py | synthetic | development_validation | yes |
| C005 | DATA_QUALITY: recall | 0.8229305002 | fraction | reports/tuning/robustness.json | required_scenario_category_recall.DATA_QUALITY.recall_per_class.NORMAL | submit/build/make_submit.py | synthetic | development_validation | yes |
| C006 | DATA_QUALITY: events | 251 | events | reports/tuning/robustness.json | required_scenario_category_recall.DATA_QUALITY.events | submit/build/make_submit.py | synthetic | development_validation | yes |
| C007 | DATA_QUALITY: recall | 0 | fraction | reports/tuning/robustness.json | required_scenario_category_recall.DATA_QUALITY.recall_per_class.WARNING | submit/build/make_submit.py | synthetic | development_validation | yes |
| C008 | DATA_QUALITY: recall | 0 | fraction | reports/tuning/robustness.json | required_scenario_category_recall.DATA_QUALITY.recall_per_class.CRITICAL | submit/build/make_submit.py | synthetic | development_validation | yes |
| C009 | COMMUNICATION_FAILURE: recall | 0.6868686869 | fraction | reports/tuning/robustness.json | required_scenario_category_recall.COMMUNICATION_FAILURE.recall_per_class.NORMAL | submit/build/make_submit.py | synthetic | development_validation | yes |
| C010 | COMMUNICATION_FAILURE: events | 132 | events | reports/tuning/robustness.json | required_scenario_category_recall.COMMUNICATION_FAILURE.events | submit/build/make_submit.py | synthetic | development_validation | yes |
| C011 | COMMUNICATION_FAILURE: recall | 0 | fraction | reports/tuning/robustness.json | required_scenario_category_recall.COMMUNICATION_FAILURE.recall_per_class.WARNING | submit/build/make_submit.py | synthetic | development_validation | yes |
| C012 | COMMUNICATION_FAILURE: recall | 0 | fraction | reports/tuning/robustness.json | required_scenario_category_recall.COMMUNICATION_FAILURE.recall_per_class.CRITICAL | submit/build/make_submit.py | synthetic | development_validation | yes |
| C013 | noise_injected: recall | 0.7879705318 | fraction | reports/tuning/robustness.json | required_scenario_category_recall.noise_injected.recall_per_class.NORMAL | submit/build/make_submit.py | synthetic | development_validation | yes |
| C014 | noise_injected: events | 1101 | events | reports/tuning/robustness.json | required_scenario_category_recall.noise_injected.events | submit/build/make_submit.py | synthetic | development_validation | yes |
| C015 | noise_injected: recall | 0 | fraction | reports/tuning/robustness.json | required_scenario_category_recall.noise_injected.recall_per_class.WARNING | submit/build/make_submit.py | synthetic | development_validation | yes |
| C016 | noise_injected: recall | 0 | fraction | reports/tuning/robustness.json | required_scenario_category_recall.noise_injected.recall_per_class.CRITICAL | submit/build/make_submit.py | synthetic | development_validation | yes |
| C017 | Selected TCN: Normalized MSE | 0.001488100946 | normalized_mse | reports/tuning/forecaster_study.json | best.mean_normalized_mse | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C018 | Persistence: Normalized MSE | 0.03728421902 | normalized_mse | reports/tuning/forecaster_study.json | best.mean_persistence_normalized_mse | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C019 | Threshold rule: Macro PR-AUC | 0.3702730284 | fraction | reports/tuning/baselines.json | models.threshold_rule.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C020 | Threshold rule: Macro PR-AUC | 0.002079029883 | fraction | reports/tuning/baselines.json | models.threshold_rule.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C021 | Threshold rule: Critical recall | 0.05809932977 | fraction | reports/tuning/baselines.json | models.threshold_rule.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C022 | Threshold rule: Critical recall | 0.01265660121 | fraction | reports/tuning/baselines.json | models.threshold_rule.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C023 | Threshold rule: Macro F1 | 0.4111786467 | fraction | reports/tuning/baselines.json | models.threshold_rule.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C024 | Threshold rule: Macro F1 | 0.004560405606 | fraction | reports/tuning/baselines.json | models.threshold_rule.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C025 | Threshold rule: Normal false-alarm rate | 0.2031290216 | fraction | reports/tuning/baselines.json | models.threshold_rule.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C026 | Threshold rule: Normal false-alarm rate | 0.001277754199 | fraction | reports/tuning/baselines.json | models.threshold_rule.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C027 | Logistic regression (C=10): Macro PR-AUC | 0.5568042896 | fraction | reports/tuning/baselines.json | models.logistic.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C028 | Logistic regression (C=10): Macro PR-AUC | 0.01821134449 | fraction | reports/tuning/baselines.json | models.logistic.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C029 | Logistic regression (C=10): Critical recall | 0.6395982451 | fraction | reports/tuning/baselines.json | models.logistic.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C030 | Logistic regression (C=10): Critical recall | 0.08165040124 | fraction | reports/tuning/baselines.json | models.logistic.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C031 | Logistic regression (C=10): Macro F1 | 0.4855133164 | fraction | reports/tuning/baselines.json | models.logistic.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C032 | Logistic regression (C=10): Macro F1 | 0.00174589501 | fraction | reports/tuning/baselines.json | models.logistic.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C033 | Logistic regression (C=10): Normal false-alarm rate | 0.3845903477 | fraction | reports/tuning/baselines.json | models.logistic.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C034 | Logistic regression (C=10): Normal false-alarm rate | 0.01039220273 | fraction | reports/tuning/baselines.json | models.logistic.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C035 | Default XGBoost: Macro PR-AUC | 0.9008595782 | fraction | reports/tuning/baselines.json | models.xgboost_default.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C036 | Default XGBoost: Macro PR-AUC | 0.008259128956 | fraction | reports/tuning/baselines.json | models.xgboost_default.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C037 | Default XGBoost: Critical recall | 0.6626078272 | fraction | reports/tuning/baselines.json | models.xgboost_default.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C038 | Default XGBoost: Critical recall | 0.05481313739 | fraction | reports/tuning/baselines.json | models.xgboost_default.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C039 | Default XGBoost: Macro F1 | 0.8297355355 | fraction | reports/tuning/baselines.json | models.xgboost_default.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C040 | Default XGBoost: Macro F1 | 0.008465652479 | fraction | reports/tuning/baselines.json | models.xgboost_default.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C041 | Default XGBoost: Normal false-alarm rate | 0.02154651743 | fraction | reports/tuning/baselines.json | models.xgboost_default.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C042 | Default XGBoost: Normal false-alarm rate | 0.001572272373 | fraction | reports/tuning/baselines.json | models.xgboost_default.folds | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C043 | Tuned XGBoost: Macro PR-AUC | 0.8907758417 | fraction | reports/tuning/tuned_cv_metrics.json | risk_models.xgboost_tuned.mean.pr_auc_macro | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C044 | Tuned XGBoost: Macro PR-AUC | 0.006443926255 | fraction | reports/tuning/tuned_cv_metrics.json | risk_models.xgboost_tuned.mean.pr_auc_macro | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C045 | Tuned XGBoost: Critical recall | 0.8170133644 | fraction | reports/tuning/tuned_cv_metrics.json | risk_models.xgboost_tuned.mean.recall_critical | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C046 | Tuned XGBoost: Critical recall | 0.04711427706 | fraction | reports/tuning/tuned_cv_metrics.json | risk_models.xgboost_tuned.mean.recall_critical | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C047 | Tuned XGBoost: Macro F1 | 0.7259906448 | fraction | reports/tuning/tuned_cv_metrics.json | risk_models.xgboost_tuned.mean.f1_macro | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C048 | Tuned XGBoost: Macro F1 | 0.00637579769 | fraction | reports/tuning/tuned_cv_metrics.json | risk_models.xgboost_tuned.mean.f1_macro | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C049 | Tuned XGBoost: Normal false-alarm rate | 0.1608714286 | fraction | reports/tuning/tuned_cv_metrics.json | risk_models.xgboost_tuned.mean.false_alarm_rate_normal | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C050 | Tuned XGBoost: Normal false-alarm rate | 0.004979786052 | fraction | reports/tuning/tuned_cv_metrics.json | risk_models.xgboost_tuned.mean.false_alarm_rate_normal | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C051 | table value: mean_abs_shap | 1.115539074 | mean_abs_shap | reports/tuning/robustness.json | tree_shap.top3.0.mean_abs_shap | submit/build/make_submit.py | synthetic | development_validation | yes |
| C052 | table value: mean_abs_shap | 0.6957067251 | mean_abs_shap | reports/tuning/robustness.json | tree_shap.top3.1.mean_abs_shap | submit/build/make_submit.py | synthetic | development_validation | yes |
| C053 | table value: mean_abs_shap | 0.5179250836 | mean_abs_shap | reports/tuning/robustness.json | tree_shap.top3.2.mean_abs_shap | submit/build/make_submit.py | synthetic | development_validation | yes |
| C054 | False alert episodes per normal event day: False alert episodes per normal event day | 0.0460797696 | episodes/day | reports/final_eval.json | alert_engine.false_alarms_per_normal_event_day | submit/build/make_submit.py | synthetic | n/a | yes |
| C055 | Median lead time: Median lead time | -3.3334 | hours | reports/final_eval.json | alert_engine.median_lead_time_hours | submit/build/make_submit.py | synthetic | n/a | yes |
| C056 | P10 lead time: P10 lead time | -11.6667 | hours | reports/final_eval.json | alert_engine.p10_lead_time_hours | submit/build/make_submit.py | synthetic | n/a | yes |
| C057 | table value: macro_pr_auc | 0.9312174331 | fraction | reports/tuning/robustness.json | learning_curve.full_train.validation_metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C058 | table value: critical_recall | 0.9301075269 | fraction | reports/tuning/robustness.json | learning_curve.full_train.validation_metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C059 | table value: macro_f1 | 0.7191591546 | fraction | reports/tuning/robustness.json | learning_curve.full_train.validation_metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C060 | table value: macro_pr_auc | 0.9319298004 | fraction | reports/tuning/robustness.json | feature_group_ablations.A_physical.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C061 | table value: critical_recall | 1 | fraction | reports/tuning/robustness.json | feature_group_ablations.A_physical.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C062 | table value: macro_f1 | 0.603365387 | fraction | reports/tuning/robustness.json | feature_group_ablations.A_physical.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C063 | table value: macro_pr_auc | 0.913456096 | fraction | reports/tuning/robustness.json | feature_group_ablations.B_temporal.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C064 | table value: critical_recall | 1 | fraction | reports/tuning/robustness.json | feature_group_ablations.B_temporal.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C065 | table value: macro_f1 | 0.5822779644 | fraction | reports/tuning/robustness.json | feature_group_ablations.B_temporal.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C066 | table value: macro_pr_auc | 0.9317587759 | fraction | reports/tuning/robustness.json | feature_group_ablations.D_vibration.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C067 | table value: critical_recall | 1 | fraction | reports/tuning/robustness.json | feature_group_ablations.D_vibration.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C068 | table value: macro_f1 | 0.6178890003 | fraction | reports/tuning/robustness.json | feature_group_ablations.D_vibration.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C069 | table value: macro_pr_auc | 0.9215787011 | fraction | reports/tuning/robustness.json | feature_group_ablations.E_sensor_health.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C070 | table value: critical_recall | 1 | fraction | reports/tuning/robustness.json | feature_group_ablations.E_sensor_health.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C071 | table value: macro_f1 | 0.5903847834 | fraction | reports/tuning/robustness.json | feature_group_ablations.E_sensor_health.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C072 | table value: macro_pr_auc | 0.8623946821 | fraction | reports/tuning/robustness.json | feature_group_ablations.F_physics.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C073 | table value: critical_recall | 0.9982078853 | fraction | reports/tuning/robustness.json | feature_group_ablations.F_physics.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C074 | table value: macro_f1 | 0.5039832556 | fraction | reports/tuning/robustness.json | feature_group_ablations.F_physics.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C075 | table value: macro_pr_auc | 0.9297280016 | fraction | reports/tuning/robustness.json | feature_group_ablations.G_dgps.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C076 | table value: critical_recall | 1 | fraction | reports/tuning/robustness.json | feature_group_ablations.G_dgps.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C077 | table value: macro_f1 | 0.6139303241 | fraction | reports/tuning/robustness.json | feature_group_ablations.G_dgps.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C078 | table value: macro_pr_auc | 0.9307847739 | fraction | reports/tuning/robustness.json | feature_group_ablations.I_terrain.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C079 | table value: critical_recall | 1 | fraction | reports/tuning/robustness.json | feature_group_ablations.I_terrain.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C080 | table value: macro_f1 | 0.6158924924 | fraction | reports/tuning/robustness.json | feature_group_ablations.I_terrain.metrics | submit/build/make_submit.py | synthetic | development_validation | yes |
| C081 | Tuned XGBoost: precision | 0.6853658537 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C082 | Tuned XGBoost: recall | 0.8492444444 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C083 | Tuned XGBoost: f1 | 0.7585549821 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C084 | Tuned XGBoost: support | 5625 | windows | reports/final_eval.json | risk_models.tuned_xgboost.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C085 | Tuned XGBoost: precision | 0.3772439949 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C086 | Tuned XGBoost: recall | 0.5304888889 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C087 | Tuned XGBoost: f1 | 0.4409309198 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C088 | Tuned XGBoost: support | 5625 | windows | reports/final_eval.json | risk_models.tuned_xgboost.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C089 | Tuned XGBoost: precision | 0.2631578947 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C090 | Tuned XGBoost: recall | 0.09333333333 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C091 | Tuned XGBoost: f1 | 0.1377952756 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C092 | Tuned XGBoost: support | 5625 | windows | reports/final_eval.json | risk_models.tuned_xgboost.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C093 | Default XGBoost: precision | 0.653159672 | fraction | reports/final_eval.json | risk_models.default_xgboost.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C094 | Default XGBoost: recall | 0.9628444444 | fraction | reports/final_eval.json | risk_models.default_xgboost.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C095 | Default XGBoost: f1 | 0.7783286628 | fraction | reports/final_eval.json | risk_models.default_xgboost.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C096 | Default XGBoost: support | 5625 | windows | reports/final_eval.json | risk_models.default_xgboost.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C097 | Default XGBoost: precision | 0.4185892336 | fraction | reports/final_eval.json | risk_models.default_xgboost.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C098 | Default XGBoost: recall | 0.5612444444 | fraction | reports/final_eval.json | risk_models.default_xgboost.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C099 | Default XGBoost: f1 | 0.4795321637 | fraction | reports/final_eval.json | risk_models.default_xgboost.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C100 | Default XGBoost: support | 5625 | windows | reports/final_eval.json | risk_models.default_xgboost.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C101 | Default XGBoost: precision | 0.2468780019 | fraction | reports/final_eval.json | risk_models.default_xgboost.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C102 | Default XGBoost: recall | 0.04568888889 | fraction | reports/final_eval.json | risk_models.default_xgboost.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C103 | Default XGBoost: f1 | 0.07710771077 | fraction | reports/final_eval.json | risk_models.default_xgboost.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C104 | Default XGBoost: support | 5625 | windows | reports/final_eval.json | risk_models.default_xgboost.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C105 | Logistic regression: precision | 0.6758842444 | fraction | reports/final_eval.json | risk_models.logistic_regression.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C106 | Logistic regression: recall | 0.3736888889 | fraction | reports/final_eval.json | risk_models.logistic_regression.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C107 | Logistic regression: f1 | 0.4812821981 | fraction | reports/final_eval.json | risk_models.logistic_regression.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C108 | Logistic regression: support | 5625 | windows | reports/final_eval.json | risk_models.logistic_regression.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C109 | Logistic regression: precision | 0.2236024845 | fraction | reports/final_eval.json | risk_models.logistic_regression.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C110 | Logistic regression: recall | 0.1984 | fraction | reports/final_eval.json | risk_models.logistic_regression.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C111 | Logistic regression: f1 | 0.2102486812 | fraction | reports/final_eval.json | risk_models.logistic_regression.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C112 | Logistic regression: support | 5625 | windows | reports/final_eval.json | risk_models.logistic_regression.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C113 | Logistic regression: precision | 0.4904262594 | fraction | reports/final_eval.json | risk_models.logistic_regression.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C114 | Logistic regression: recall | 0.7649777778 | fraction | reports/final_eval.json | risk_models.logistic_regression.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C115 | Logistic regression: f1 | 0.5976803945 | fraction | reports/final_eval.json | risk_models.logistic_regression.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C116 | Logistic regression: support | 5625 | windows | reports/final_eval.json | risk_models.logistic_regression.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C117 | Threshold rule: precision | 0.5712661499 | fraction | reports/final_eval.json | risk_models.threshold_rule.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C118 | Threshold rule: recall | 0.9825777778 | fraction | reports/final_eval.json | risk_models.threshold_rule.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C119 | Threshold rule: f1 | 0.7224836601 | fraction | reports/final_eval.json | risk_models.threshold_rule.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C120 | Threshold rule: support | 5625 | windows | reports/final_eval.json | risk_models.threshold_rule.per_class.NORMAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C121 | Threshold rule: precision | 0.3786441869 | fraction | reports/final_eval.json | risk_models.threshold_rule.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C122 | Threshold rule: recall | 0.3832888889 | fraction | reports/final_eval.json | risk_models.threshold_rule.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C123 | Threshold rule: f1 | 0.380952381 | fraction | reports/final_eval.json | risk_models.threshold_rule.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C124 | Threshold rule: support | 5625 | windows | reports/final_eval.json | risk_models.threshold_rule.per_class.WARNING | submit/build/make_submit.py | synthetic | locked_test | yes |
| C125 | Threshold rule: precision | 0.1553784861 | fraction | reports/final_eval.json | risk_models.threshold_rule.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C126 | Threshold rule: recall | 0.0416 | fraction | reports/final_eval.json | risk_models.threshold_rule.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C127 | Threshold rule: f1 | 0.06562894405 | fraction | reports/final_eval.json | risk_models.threshold_rule.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C128 | Threshold rule: support | 5625 | windows | reports/final_eval.json | risk_models.threshold_rule.per_class.CRITICAL | submit/build/make_submit.py | synthetic | locked_test | yes |
| C129 | table value: count | 4777 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.confusion_matrix[0][0] | submit/build/make_submit.py | synthetic | locked_test | yes |
| C130 | table value: count | 691 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.confusion_matrix[0][1] | submit/build/make_submit.py | synthetic | locked_test | yes |
| C131 | table value: count | 157 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.confusion_matrix[0][2] | submit/build/make_submit.py | synthetic | locked_test | yes |
| C132 | table value: count | 1328 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.confusion_matrix[1][0] | submit/build/make_submit.py | synthetic | locked_test | yes |
| C133 | table value: count | 2984 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.confusion_matrix[1][1] | submit/build/make_submit.py | synthetic | locked_test | yes |
| C134 | table value: count | 1313 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.confusion_matrix[1][2] | submit/build/make_submit.py | synthetic | locked_test | yes |
| C135 | table value: count | 865 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.confusion_matrix[2][0] | submit/build/make_submit.py | synthetic | locked_test | yes |
| C136 | table value: count | 4235 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.confusion_matrix[2][1] | submit/build/make_submit.py | synthetic | locked_test | yes |
| C137 | table value: count | 525 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.confusion_matrix[2][2] | submit/build/make_submit.py | synthetic | locked_test | yes |
| C138 | Generated event sequences: value | 10000 | events | data/synthetic/dataset_manifest.json | row_counts.sequences_generated | submit/build/make_submit.py | synthetic | synthetic_corpus | yes |
| C139 | Generated sensor observations: value | 1440000 | rows | data/synthetic/dataset_manifest.json | row_counts.synthetic_nodes_rows | submit/build/make_submit.py | synthetic | synthetic_corpus | yes |
| C140 | Post-fix feature windows: value | 90000 | windows | data/synthetic/dataset_manifest.json | feature_store.n_windows | submit/build/make_submit.py | synthetic | synthetic_corpus | yes |
| C141 | Window length: value | 60 | timesteps | data/synthetic/dataset_manifest.json | feature_store.window_steps | submit/build/make_submit.py | synthetic | synthetic_corpus | yes |
| C142 | Window stride: value | 10 | timesteps | data/synthetic/dataset_manifest.json | feature_store.stride_steps | submit/build/make_submit.py | synthetic | synthetic_corpus | yes |
| C143 | Scenario families: value | 16 | families | data/synthetic/dataset_manifest.json | scenario_parameters.scenario_types | submit/build/make_submit.py | synthetic | synthetic_corpus | yes |
| C144 | Development training events: value | 6516 | events | data/synthetic/dataset_manifest.json | split_definition.event_counts.train | submit/build/make_submit.py | synthetic | train | yes |
| C145 | Development validation events: value | 1609 | events | data/synthetic/dataset_manifest.json | split_definition.event_counts.validation | submit/build/make_submit.py | synthetic | validation | yes |
| C146 | Fresh locked test events: value | 1875 | events | reports/final_eval.json | evaluation.events | submit/build/make_submit.py | synthetic | locked_test | yes |
| C147 | Fresh locked test windows: value | 16875 | windows | reports/final_eval.json | evaluation.windows | submit/build/make_submit.py | synthetic | locked_test | yes |
| C148 | Locked test seed: value | 20260929 | seed | reports/test_lock.json | seed | submit/build/make_submit.py | synthetic | locked_test | yes |
| C149 | Locked test evaluations used: value | 1 | evaluations | reports/test_lock.json | evals_run | submit/build/make_submit.py | synthetic | locked_test | yes |
| C150 | Tuned XGBoost active inputs: value | 33 | features | reports/tuning/tuned_cv_metrics.json | protocol.feature_count | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C151 | Tuned IF active inputs: value | 33 | features | configs/model_params.yaml | isolation_forest.features | submit/build/make_submit.py | synthetic | train | yes |
| C152 | Locked test NORMAL support: value | 5625 | windows | reports/final_eval.json | risk_models.tuned_xgboost.per_class.NORMAL.support | submit/build/make_submit.py | synthetic | locked_test | yes |
| C153 | Locked test WARNING support: value | 5625 | windows | reports/final_eval.json | risk_models.tuned_xgboost.per_class.WARNING.support | submit/build/make_submit.py | synthetic | locked_test | yes |
| C154 | Locked test CRITICAL support: value | 5625 | windows | reports/final_eval.json | risk_models.tuned_xgboost.per_class.CRITICAL.support | submit/build/make_submit.py | synthetic | locked_test | yes |
| C155 | Nodes per event in current corpus: value | 1 | node/event | reports/acceptance_criteria.md | G-5: one node per event | submit/build/make_submit.py | synthetic | synthetic_corpus | yes |
| C156 | Tuned XGBoost: Macro PR-AUC | 0.5028714642 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.pr_auc_macro_ovr | submit/build/make_submit.py | synthetic | locked_test | yes |
| C157 | Tuned XGBoost: Macro F1 | 0.4457603925 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.f1_macro | submit/build/make_submit.py | synthetic | locked_test | yes |
| C158 | Tuned XGBoost: Critical recall | 0.09333333333 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.recall_critical | submit/build/make_submit.py | synthetic | locked_test | yes |
| C159 | Tuned XGBoost: Normal false-alarm rate | 0.1507555556 | fraction | reports/final_eval.json | risk_models.tuned_xgboost.false_alarm_rate_normal | submit/build/make_submit.py | synthetic | locked_test | yes |
| C160 | Default XGBoost: Macro PR-AUC | 0.5351496371 | fraction | reports/final_eval.json | risk_models.default_xgboost.pr_auc_macro_ovr | submit/build/make_submit.py | synthetic | locked_test | yes |
| C161 | Default XGBoost: Macro F1 | 0.4449895124 | fraction | reports/final_eval.json | risk_models.default_xgboost.f1_macro | submit/build/make_submit.py | synthetic | locked_test | yes |
| C162 | Default XGBoost: Critical recall | 0.04568888889 | fraction | reports/final_eval.json | risk_models.default_xgboost.recall_critical | submit/build/make_submit.py | synthetic | locked_test | yes |
| C163 | Default XGBoost: Normal false-alarm rate | 0.03715555556 | fraction | reports/final_eval.json | risk_models.default_xgboost.false_alarm_rate_normal | submit/build/make_submit.py | synthetic | locked_test | yes |
| C164 | Logistic regression: Macro PR-AUC | 0.4687112475 | fraction | reports/final_eval.json | risk_models.logistic_regression.pr_auc_macro_ovr | submit/build/make_submit.py | synthetic | locked_test | yes |
| C165 | Logistic regression: Macro F1 | 0.4297370913 | fraction | reports/final_eval.json | risk_models.logistic_regression.f1_macro | submit/build/make_submit.py | synthetic | locked_test | yes |
| C166 | Logistic regression: Critical recall | 0.7649777778 | fraction | reports/final_eval.json | risk_models.logistic_regression.recall_critical | submit/build/make_submit.py | synthetic | locked_test | yes |
| C167 | Logistic regression: Normal false-alarm rate | 0.6263111111 | fraction | reports/final_eval.json | risk_models.logistic_regression.false_alarm_rate_normal | submit/build/make_submit.py | synthetic | locked_test | yes |
| C168 | Threshold rule: Macro PR-AUC | 0.4145839077 | fraction | reports/final_eval.json | risk_models.threshold_rule.pr_auc_macro_ovr | submit/build/make_submit.py | synthetic | locked_test | yes |
| C169 | Threshold rule: Macro F1 | 0.3896883284 | fraction | reports/final_eval.json | risk_models.threshold_rule.f1_macro | submit/build/make_submit.py | synthetic | locked_test | yes |
| C170 | Threshold rule: Critical recall | 0.0416 | fraction | reports/final_eval.json | risk_models.threshold_rule.recall_critical | submit/build/make_submit.py | synthetic | locked_test | yes |
| C171 | Threshold rule: Normal false-alarm rate | 0.01742222222 | fraction | reports/final_eval.json | risk_models.threshold_rule.false_alarm_rate_normal | submit/build/make_submit.py | synthetic | locked_test | yes |
| C172 | risk_xgboost_tuned: p50_ms | 0.3170205 | ms/window | reports/tuning/robustness.json | workstation_edge_profile.models.risk_xgboost_tuned | submit/build/make_submit.py | synthetic | n/a | yes |
| C173 | risk_xgboost_tuned: p95_ms | 0.6130708 | ms/window | reports/tuning/robustness.json | workstation_edge_profile.models.risk_xgboost_tuned | submit/build/make_submit.py | synthetic | n/a | yes |
| C174 | risk_xgboost_tuned: peak_sampled_rss_mib | 550.625 | MiB | reports/tuning/robustness.json | workstation_edge_profile.models.risk_xgboost_tuned | submit/build/make_submit.py | synthetic | n/a | yes |
| C175 | risk_xgboost_tuned: artifact_bytes | 6498680 | bytes | reports/tuning/robustness.json | workstation_edge_profile.models.risk_xgboost_tuned | submit/build/make_submit.py | synthetic | n/a | yes |
| C176 | isolation_forest_tuned: p50_ms | 2.9885 | ms/window | reports/tuning/robustness.json | workstation_edge_profile.models.isolation_forest_tuned | submit/build/make_submit.py | synthetic | n/a | yes |
| C177 | isolation_forest_tuned: p95_ms | 3.8287875 | ms/window | reports/tuning/robustness.json | workstation_edge_profile.models.isolation_forest_tuned | submit/build/make_submit.py | synthetic | n/a | yes |
| C178 | isolation_forest_tuned: peak_sampled_rss_mib | 501.359375 | MiB | reports/tuning/robustness.json | workstation_edge_profile.models.isolation_forest_tuned | submit/build/make_submit.py | synthetic | n/a | yes |
| C179 | isolation_forest_tuned: artifact_bytes | 49315608 | bytes | reports/tuning/robustness.json | workstation_edge_profile.models.isolation_forest_tuned | submit/build/make_submit.py | synthetic | n/a | yes |
| C180 | forecaster_tuned: p50_ms | 0.041708 | ms/window | reports/tuning/robustness.json | workstation_edge_profile.models.forecaster_tuned | submit/build/make_submit.py | synthetic | n/a | yes |
| C181 | forecaster_tuned: p95_ms | 0.04479315 | ms/window | reports/tuning/robustness.json | workstation_edge_profile.models.forecaster_tuned | submit/build/make_submit.py | synthetic | n/a | yes |
| C182 | forecaster_tuned: peak_sampled_rss_mib | 627.1875 | MiB | reports/tuning/robustness.json | workstation_edge_profile.models.forecaster_tuned | submit/build/make_submit.py | synthetic | n/a | yes |
| C183 | forecaster_tuned: artifact_bytes | 32587 | bytes | reports/tuning/robustness.json | workstation_edge_profile.models.forecaster_tuned | submit/build/make_submit.py | synthetic | n/a | yes |
| C184 | Maximum subsidence from extraction geometry | 2100 | mm | configs/physics.yaml | physics.extraction_height * physics.subsidence_factor * 1000 | submit/build/make_submit.py | synthetic | n/a | yes |
| C185 | Knothe temporal coefficient | 0.25 | 1/day | configs/physics.yaml | physics.time_coefficient | submit/build/make_submit.py | synthetic | n/a | yes |
| C186 | Panel influence radius | 120 | m | configs/physics.yaml | physics.influence_radius | submit/build/make_submit.py | synthetic | n/a | yes |
| C187 | Panel centre x | 0 | m | configs/physics.yaml | physics.panel_center_x | submit/build/make_submit.py | synthetic | n/a | yes |
| C188 | Panel centre y | 0 | m | configs/physics.yaml | physics.panel_center_y | submit/build/make_submit.py | synthetic | n/a | yes |
| C189 | Physics profile time point | 3 | days | submit/build/make_submit.py | t_days | submit/build/make_submit.py | synthetic | n/a | yes |
| C190 | Pipeline stage count | 8 | stages | src/pipeline.py | validation through explainability | submit/build/make_submit.py | synthetic | n/a | yes |
| C191 | XGBoost search requested trials | 40 | trials | reports/tuning/xgboost_study.json | study.requested_trials | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C192 | XGBoost search actual trials | 16 | trials | reports/tuning/xgboost_study.json | study.actual_trials | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C193 | XGBoost search completed trials | 12 | trials | reports/tuning/xgboost_study.json | study.completed_trials | submit/build/make_submit.py | synthetic | grouped_cv | yes |
| C194 | XGBoost search pruned trials | 4 | trials | reports/tuning/xgboost_study.json | study.pruned_trials | submit/build/make_submit.py | synthetic | grouped_cv | yes |
