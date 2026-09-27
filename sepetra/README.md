# Sepetra — Model Evaluation Report

§15 risk model (XGBoost) evaluated on the **test split** (18,000 windows, 2,000 events, T-068 event-level splits). Alert rule: P(CRITICAL) ≥ 0.5 (same convention as the T-070 ablation, so numbers match `experiments/ablation_a_to_f.json`).

## Headline metrics

| metric | value |
|---|---|
| accuracy | 0.890 |
| precision (binary alert) | 0.361 |
| recall (binary alert) | 0.443 |
| F1 (binary alert) | 0.398 |
| precision (macro) | 0.881 |
| recall (macro) | 0.797 |
| F1 (macro) | 0.833 |
| PR-AUC (CRITICAL vs rest) | 0.901 |
| ROC-AUC (CRITICAL vs rest) | 0.982 |
| Brier (P-CRITICAL) | 0.0558 |
| ECE (P-CRITICAL) | 0.0023 |

**Why two precisions?** The *binary* view scores the operational question “did the mesh raise an alert on a disturbed window?”; the *macro* view scores the 3-class risk grading. Both are reported — accuracy alone would flatter a model that never warns (NORMAL dominates this mesh), which is why the repo's §24 headline set omits it while this report shows it with context.

## Per-class detail

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| CRITICAL | 1,881 | 0.916 | 0.704 | 0.796 |
| NORMAL | 12,375 | 0.902 | 0.971 | 0.935 |
| WARNING | 3,744 | 0.826 | 0.715 | 0.766 |

## Graphs (`graphs/`)

| file | what it shows |
|---|---|
| confusion_matrices.png | counts + row-normalised confusion matrices |
| roc_curves.png | one-vs-rest ROC with AUC per class |
| precision_recall_curves.png | one-vs-rest precision-recall (imbalance-honest) |
| metric_bars.png | headline accuracy/precision/recall/F1 side by side |
| calibration_curve.png | predicted vs observed frequency for P(CRITICAL) |
| per_class_metrics.png | precision/recall/F1 per class with support |
| lead_time_false_alarm.png | operational view: lead time + false alarms per event |

## Operational summary

- Median lead time on alerted events: 0.6 h (P10 0.0 h).
- Events alerted: 174 of 2000.
- Mean false-alarm windows per event: 0.29.

## Reproduce

```bash
.venv/bin/python sepetra/build_report.py
```

Deterministic: config-driven hyperparameters, fixed T-068 splits, no sampling.
Per-window scores: `predictions_test.parquet`. Machine-readable metrics: `metrics.json`.

## Honest reading

- **Accuracy is high but weakly informative here**: NORMAL is the majority class (12,375 of 18,000 test windows), so the §24 discipline prefers precision/recall/F1 and calibration as headline numbers.
- The binary alert view trades precision for recall at the 0.5 threshold — see `precision_recall_curves.png` for the full trade-off surface before moving it.
- Lead time is bounded by the §10 event generator (onset mid-event); see the T-070 gate discussion in `reports/ablation.md`.
