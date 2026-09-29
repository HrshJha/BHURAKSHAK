#!/usr/bin/env python3
"""Render an aggregate-only optimization report; never opens test examples."""
import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import yaml


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default='reports/optimization');args=parser.parse_args()
    out=ROOT/args.output_dir
    cfg=yaml.safe_load((out/'best_development_config.yaml').read_text())
    cv=pd.read_csv(out/'model_comparison.csv');test=json.loads((out/'independent_metrics.json').read_text())
    operating=json.loads((out/'operating_points.json').read_text());latency=json.loads((out/'latency.json').read_text())
    def risk_table(scope):
        rows=cv[(cv.scope==scope)&cv.critical_recall.notna()]
        lines=['| Model | Critical recall | Normal FPR | Macro AP | Macro F1 | Brier | ECE |','|---|---:|---:|---:|---:|---:|---:|']
        for _,r in rows.iterrows():lines.append(f'| {r.experiment} | {100*r.critical_recall:.2f}% | {100*r.normal_fpr:.2f}% | {r.macro_pr_auc:.4f} | {r.macro_f1:.4f} | {r.brier:.4f} | {r.ece:.4f} |')
        return '\n'.join(lines)
    def if_table(scope):
        rows=cv[(cv.scope==scope)&cv.anomaly_recall.notna()]
        lines=['| Model | Anomaly recall | Anomaly FPR | Anomaly AP | Macro F1 |','|---|---:|---:|---:|---:|']
        for _,r in rows.iterrows():lines.append(f'| {r.experiment} | {100*r.anomaly_recall:.2f}% | {100*r.anomaly_fpr:.2f}% | {r.anomaly_pr_auc:.4f} | {r.anomaly_macro_f1:.4f} |')
        return '\n'.join(lines)
    thresholds='\n'.join(f"| {100*float(r['fit_fpr_cap']):.0f}% | {100*r['critical_recall']:.2f}% | {100*r['normal_fpr']:.2f}% | {r['macro_f1']:.4f} |" for r in operating)
    timing='\n'.join(f"| {r['model']} | {r['median_ms']:.3f} | {r['p95_ms']:.3f} |" for r in latency)
    tests=(out/'full_tests.log').read_text().strip().splitlines()[-1] if (out/'full_tests.log').exists() else 'not rerun in this reproduction directory; run the documented pytest command'
    report=f'''# BHURAKSHAK bounded optimization report

The candidate improved unseen-scenario Critical recall but **did not meet the acceptance target**. On the same new synthetic test, the exact original frozen XGBoost artifact achieved 10.67% Critical recall / 13.11% Normal FPR; the candidate achieved **29.22% / 7.44%**. Required: >=89% / <=5%. The candidate remains experimental; deployed models and previous evidence were preserved. No field effectiveness or collapse-time prediction is demonstrated.

## Protocol and completed work

- Original data: v2 label-blind feature store. Only original train/validation events are materialized. Original locked samples and predictions were not read; their published aggregate report was used for context only.
- Selection: 58,644 train windows, three folds grouped by `generation_parameter_id`. Each outer training fold is split approximately 60/20/20 into fitting, calibration and threshold groups. Whole groups stay together. The 14,481 original validation windows are excluded from this run's selection; historical model parameters had already used that validation set.
- Fixed seed 42, one model thread. Exactly 20 Optuna attempts per model: XGBoost 17 completed / 3 pruned; Isolation Forest 11 completed / 9 pruned; zero failed trials. Seeded TPE and median pruning after two folds, with five startup trials. SQLite checkpoints and every trial's parameters/partial folds are saved. A fresh uninterrupted run is deterministic in this environment; resumed TPE execution need not produce the identical future suggestion sequence.
- Baseline refits use the original hyperparameters/features on the **same inner partitions and strict decision protocol** as candidates. They isolate model/feature changes from protocol changes. The exact old frozen artifacts are also compared in the independent evaluation, retaining their original calibration and threshold semantics.
- Candidates test displacement/health versus displacement/health/configured-physics features. Tilt, strain and vibration are quarantined because of simulator defects described in `audit.md`. No labels or existing data were changed. Gated spatial/DGPS/InSAR inputs remain excluded.
- Weight search: none/square-root/inverse class-frequency weighting, plus Critical cost multiplier 1–4. Calibration choices: none/sigmoid/isotonic. Critical-first joint Normal false-alarm budget replaces argmax-promoting overrides. The new inference contract validates features, numeric class order and probability sums.
- One fresh test specification was fixed before selection: seed 20261007, 100 events each of stable/accelerating/rapid scenarios, 300 events / 2,700 windows. Generation occurred only after model/config hashing. The one-use lock was claimed first, and a second invocation was refused before data access. This is a same-simulator unseen-family evaluation, not independent physical validation.

## Development results

Three-fold means below are **model-selection estimates**, not unbiased final performance. Critical recall across the candidate's folds ranged from 67.20% to 86.20%; correlated windows do not justify row-independent confidence intervals.

{risk_table('train_grouped_cv_selection')}

The physics-prior ablation reduced Critical recall to 68.60%, macro AP to 0.6736, and increased Normal FPR to 4.98%. Retain the physics subset for this experimental run. Group-block permutation importance on reserved validation identified rolling minimum as the largest measured AP contribution; this post-selection diagnostic did not change the feature list.

Reserved development validation (not used for this run's selection):

{risk_table('reserved_development_validation')}

The candidate's 92.65% recall does **not** satisfy the joint target: 5.24% Normal FPR exceeds 5%. This split is far easier than unseen rapid-subsidence events. Brier improves but ECE worsens versus the comparable baseline; calibration is not uniformly improved.

Threshold alternatives fitted on inner decision groups, then measured unchanged on reserved validation:

| Fit Normal FPR budget | Validation Critical recall | Validation Normal FPR | Macro F1 |
|---|---:|---:|---:|
{thresholds}

These are pre-fitted alternative operating points, not test-driven adjustments. `critical_recall_curve.png` is a diagnostic Critical-versus-Normal sweep and is explicitly distinct from the full Warning-or-Critical false-alarm rate.

## Independent evaluation

{risk_table('independent_synthetic_test')}

The candidate improves relative to both the exact historical artifact and the comparable refit, but neither recall nor FPR meets acceptance. No model/threshold was changed after viewing these results. The old published 9.33% XGBoost recall and 89.59% Isolation Forest anomaly recall came from a different consumed corpus and are not substituted into this same-corpus comparison.

Isolation Forest development selection:

{if_table('train_grouped_cv_selection')}

Isolation Forest independent test:

{if_table('independent_synthetic_test')}

The Isolation Forest search did **not** beat the comparable original configuration. Its 98.02% anomaly recall is not Critical-class recall, and its higher recall than the old frozen artifact uses a higher FPR. Do not promote it as an improvement. Development contains harder fault/anomaly scenarios and imperfect observable labels absent from the three-scenario final test; the large anomaly-metric shift is expected from this scope difference.

Metrics: Normal FPR = fraction of actual NORMAL rows predicted WARNING or CRITICAL. Anomaly FPR = flagged fraction of anomaly_label=0 rows, regardless of risk class. Macro PR-AUC is macro average precision (stepwise AP, not trapezoidal integration); F1 averages all classes. Brier is the mean summed three-class squared error; ECE uses 15 confidence bins. Missing/not-applicable cells remain empty.

## Alerts, robustness and latency

Twelve WATCH threshold/persistence combinations were evaluated on inner decision groups. The selected setting stayed at probability >0.5 for three consecutive windows. In the new test, WATCH occurred in 87/100 Critical events versus 68/100 for the old frozen risk model; false WATCH episodes occurred in 5/100 Normal events versus 3/100. Median first WATCH was 13.1667 hours after synthetic scenario start **among detected events**, so missed events must not be omitted from interpretation. Classifier Critical recall and WATCH-event recall are different quantities.

Only GREEN/WATCH were emitted. WARNING/CRITICAL escalation remains blocked by unavailable spatial/neighbor evidence; no additional nodes or trapdoors were invented. The existing AlertEngine lacks recovery/de-escalation implementation. Event labels are constant across time, so genuine event-onset/collapse lead time is **not measurable**. Observed-window exposure yields 0.0900 false WATCH episodes per Normal event-day for the candidate; this excludes feature-window warmup and is not a field alarm rate.

Feature-noise sensitivity (diagnostic only, not a physically coupled sensor simulation): adding 1%, 5%, 10% of training feature SD reduced validation Critical recall to 82.97%, 58.42%, 48.21%. This fragility limits deployment readiness. Synthetic features were not augmented using these perturbations.

| Model | Median one-row latency (ms) | p95 (ms) |
|---|---:|---:|
{timing}

100 warmed workstation calls. Risk timing includes pandas extraction, calibration and thresholds; IF timing is prepared-array scoring. Both exclude raw feature extraction, transport, hardware sampling and alert state processing. These are not edge-device latency measurements.

## Selected configuration and decisions

XGBoost: {cfg['risk']['n_estimators']} trees, depth {cfg['risk']['max_depth']}, learning rate {cfg['risk']['learning_rate']:.6f}, min_child_weight {cfg['risk']['min_child_weight']:.6f}, sigmoid calibration, no frequency reweighting, Critical weight {cfg['risk']['critical_weight']:.6f}; 18 displacement/health/physics features. Thresholds: Critical {cfg['thresholds']['critical']:.8f}, Warning {cfg['thresholds']['warning']:.8f}. Full parameters, feature order, data split sizes and thresholds are in `best_development_config.yaml`.

IF: {cfg['isolation_forest']['n_estimators']} trees, max_samples {cfg['isolation_forest']['max_samples']}, max_features {cfg['isolation_forest']['max_features']:.6f}, no bootstrap; threshold {cfg['anomaly_threshold']:.8f}. Retain only as an experimental searched candidate because the comparable original was better. Original deployments remain unchanged.

Retained implementation improvements: strict/grouped evaluation, explicit inference class/feature contract, bounded studies, quarantined candidate inputs, reproducible frozen bundle and single-use test guard. Rejected/deferred: feature-only ablation, claimed IF improvement, deployment promotion, binary alternative/TreeSHAP/additional synthetic augmentation under the bounded budget. No failed technical trials were hidden.

## Remaining requirements

1. Correct scenario-specific tilt/strain coupling and inject actual signals for disturbance/vibration anomaly labels before versioned multimodal regeneration. Scenario-wide Critical labels, sometimes at tiny local deformation, are simulator taxonomy rather than measured hazard severity. Do not relabel them to improve scores.
2. Correct the existing missing-endpoint velocity definition and rebuild a new feature schema; this run preserves the historical store for comparison. These observations were audited/quarantined where possible, not claimed fully repaired.
3. Acquire physically justified transition/onset and recovery data. One-trapdoor hardware scope is unchanged. Multi-node confirmation and field effectiveness remain unvalidated.
4. Broaden training coverage using independent data under an explicitly revised experiment protocol; then use another genuinely fresh locked evaluation. The consumed test in this report cannot become a tuning set.
5. Three CV folds and 100 test events/class do not establish robustness across mines, site geometries or real sensor distributions. No test-dependent follow-up search was run.

## Validation and reproduction

Full suite: **{tests}**. New tests cover strict threshold gates, joint FPR budgeting, nested parameter-group separation, canonical tuner groups, single-use locks, invalid probabilities, gated-feature refusal and inference feature order/label blindness. Existing artifact tests now use an isolated registry. Label-free CLI smoke produced nine predictions; hashes matched. The three required PNGs were visually inspected. Package versions and machine details are saved in `runtime.json`.

From repository root, using the existing label-blind feature store and split assignment:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -o addopts='' -q
.venv/bin/python scripts/report_optimization.py
```

To reproduce development experiments without overwriting this run:

```sh
.venv/bin/python scripts/optimize_bhurakshak.py --output-dir reports/optimization_reproduction
```

The historical commands for this completed run were:

```sh
.venv/bin/python scripts/optimize_bhurakshak.py
.venv/bin/python scripts/evaluate_optimization.py
```

Both now refuse to overwrite the frozen run or reuse its test. For a reproducibility audit only, an isolated fresh output directory can deterministically reconstruct the same synthetic evaluation from the predeclared specification:

```sh
.venv/bin/python scripts/evaluate_optimization.py --output-dir reports/optimization_reproduction --test-dir data/optimization_reproduction_independent_v1
.venv/bin/python scripts/report_optimization.py --output-dir reports/optimization_reproduction
```

That recreated seed is **the same consumed test**, not new independent evidence; it must never drive tuning. A future scientific evaluation requires a separately specified fresh protocol/seed. The canonical report is regenerated from saved aggregates only. Original model artifacts are required for exact historical comparisons.

Experimental inference on a label-blind, v2-compatible feature CSV/Parquet:

```sh
.venv/bin/python scripts/predict_optimized.py --features /absolute/path/features.parquet --output /absolute/path/predictions.csv
```

Do not point inference at locked evaluation data. This entry point does not replace `src/pipeline.py`'s existing default-training path.

## Important files

| Files | Purpose |
|---|---|
| `scripts/optimize_bhurakshak.py`, `configs/optimization.yaml` | Bounded training/search, folds, calibration/thresholds, diagnostics and freeze |
| `src/risk/optimized.py`, `scripts/predict_optimized.py` | Frozen experimental inference, feature/probability checks, strict decisions |
| `scripts/evaluate_optimization.py` | Fresh synthetic generation after freeze, exclusive one-use lock, paired comparisons |
| `scripts/report_optimization.py` | Aggregate-only report regeneration |
| `scripts/tune_models.py` | Development filter before materialization; canonical tuning grouped by generating parameters |
| `src/risk/xgboost_model.py` | Explicit softprob objective; invalid requested feature subsets fail closed |
| `tests/test_optimization.py`, `tests/test_artifacts_and_registry.py` | Evaluation/inference regressions; protect deployment registry from tests |
| `models/optimization/candidate.joblib` | Candidate and comparable baseline bundles, calibration, thresholds and alert setting |
| `reports/optimization/best_development_config.yaml`, `freeze.json`, `independent_test_lock.json` | Frozen configurations, model/corpus digests, one-use evaluation record |
| `reports/optimization/experiment_log.csv`, `model_comparison.csv`, `*_study.json`, `studies.sqlite3` | All 40 trial attempts, comparisons and resumable checkpoints |
| `reports/optimization/*_oof.csv`, `*_folds.json`, `fold_assignment.csv` | Development predictions and split/variance evidence |
| `reports/optimization/*.png`, `operating_points.json`, `permutation_importance.csv`, `feature_group_distributions.csv`, `scenario_errors.csv` | Requested plots, feature and scenario diagnostics |
| `reports/optimization/alert_*.json`, `robustness.json`, `latency.json`, `independent_metrics.json`, `validation_metrics.json` | Alert, robustness and separate development/test measurements |
| `reports/optimization/TASKS.md`, `audit.md`, `full_tests.log`, `runtime.json`, this report | Task disposition, defects/limitations, tests and reproduction context |
| `.gitignore` | Keep new one-use evaluation raw files out of version control |

Original reports, models, training data, labels and existing locked test files were preserved. No repository-wide data regeneration or hardware changes were performed.
'''
    (out/'final_report.md').write_text(report)
    runtime=dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost','optuna','pyarrow','joblib','matplotlib','PyYAML']})
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    print(out/'final_report.md')

if __name__=='__main__':main()
