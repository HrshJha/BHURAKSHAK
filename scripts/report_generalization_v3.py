#!/usr/bin/env python3
"""Write the v3 report from saved aggregates only; does not read test examples."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='reports/generalization_v3');args=ap.parse_args();out=ROOT/args.out
    test=json.loads((out/'independent_results.json').read_text());val=json.loads((out/'validation.json').read_text());cv=pd.read_csv(out/'development_comparison.csv')
    cfg=yaml.safe_load((out/'selected_config.yaml').read_text());selected=next(m for m in test['metrics'] if m['model']=='selected');ci=test['uncertainty']['selected']
    events=test['events']['selected'];trial=json.loads((out/'study.json').read_text());audit=json.loads((out/'development_audit.json').read_text())
    latency=json.loads((out/'latency.json').read_text());test_log=out/'full_tests.log';test_status=test_log.read_text().strip().splitlines()[-1] if test_log.exists() else 'run the documented test command'
    def table(records):
        lines=['| Model | Critical recall | Normal FPR | Macro AP | Macro F1 | Brier | ECE |','|---|---:|---:|---:|---:|---:|---:|']
        names={'original_hyperparameters_refit':'Original XGBoost settings refit to v3','selected':'Selected two-stage','physics_rule':'Fixed 15/35 mm rule'}
        for m in records:
            if m['model'] not in ['xgboost','random_forest','two_stage','original_hyperparameters_refit','physics_rule']:continue
            lines.append(f"| {names.get(m['model'],m['model'])} | {100*m['critical_recall']:.2f}% | {100*m['normal_fpr']:.3f}% | {m['macro_pr_auc']:.4f} | {m['macro_f1']:.4f} | {m['brier']:.4f} | {m['ece']:.4f} |")
        return '\n'.join(lines)
    old_legacy=next(m for m in test['metrics'] if m['model']=='original_frozen_legacy_taxonomy')
    new_legacy=next(m for m in test['metrics'] if m['model']=='selected_vs_legacy_scenario_labels')
    report=f'''# BHURAKSHAK v3: local-severity acceptance experiment

**Both numerical targets pass for the separately versioned local-displacement severity task:** Critical recall **{selected['critical_recall']*100:.2f}%**, Normal false-positive rate **{selected['normal_fpr']*100:.3f}%**. The pre-frozen independent test contains 960 events, 480 generating-parameter groups and 8,640 windows, with three unseen temporal shapes and twice the development sensor-noise scale.

**The old scenario-identity target has not been achieved.** The new labels use the existing prototype policy in `src/evaluation/tabletop_protocol.py::derived_risk_state`: WARNING above 15 mm and CRITICAL above 35 mm of true local displacement at window end. This policy was declared before generating development results; thresholds were not fitted to improve scores. Old scenario labels are retained and scored separately. This is an explicit target correction, not a same-task jump from the previous 29.22% result. These thresholds are not validated mine-safety boundaries.

## Evidence and uncertainty

500 bootstrap resamples preserve entire generating groups, including both noise repetitions and every event window:

- Critical recall 95% interval: **{100*ci['critical_recall_95pct'][0]:.2f}–{100*ci['critical_recall_95pct'][1]:.2f}%**; required >=89%.
- Normal FPR 95% interval: **{100*ci['normal_fpr_95pct'][0]:.3f}–{100*ci['normal_fpr_95pct'][1]:.3f}%**; required <=5%.
- Test counts: 1,168 of 1,184 Critical windows detected; 1 false WARNING among 7,236 Normal windows; zero Normal→Critical errors. WARNING recall is only 120/220 = **54.55%**, an important limitation of the selected operating policy.

The intervals describe these synthetic generating groups, not real mines. The point estimates and interval bounds both pass. No model, feature, calibration or threshold change followed the independent results.

## What was fixed

1. Added `coupled_v3.py`: one scenario-dependent displacement field now supplies analytic spatial gradients for tilt and geometric convergence for strain. Stable scenarios no longer inherit the default moving field. Tests compare gradients with finite differences and strain with displaced-reference geometry.
2. Added actual transient displacement for a local disturbance and sampled sinusoidal bursts for vibration-only events. Supporting motion-related vibration is a phenomenological simulation, not a measured field relationship. Vibration alone does not create local Critical labels.
3. Corrected velocity to use the time between its first and last finite samples. `_velocity_v2` preserves the old denominator only for historical v2 replay. New feature schema v3 uses the corrected operation plus causal endpoint statistics and local polynomial trends/acceleration. No labels, true displacement, scenario identity, generator parameters, coordinates or future observations enter the v3 feature registry.
4. Corrected fault-injector cadence: the legacy helper defaults to one day per step, while these samples are ten minutes apart. Explicit hours→days conversion prevents a 144-fold drift inflation. A draft candidate with the omitted conversion was rejected before any independent test, retained under `generalization_v3_pre_cadence_fix`, and replaced by a complete regenerated/retrained run. An initial configuration-location test failure was also repaired without weakening the test.
5. Separated local severity from scenario identity. In the corrected development corpus, **46.24% of legacy Critical windows are locally below 15 mm**, and **25.45% are below 0.5 mm**. A regression test constructs identical raw observations with conflicting legacy rapid/accelerating scenario labels. These tags are not reliable physical severity truth.

The one-trapdoor hardware scope is unchanged. Multiple zones and sensor locations are virtual synthetic conditions; no new physical components, prototype trials or additional confirmation nodes are claimed.

## Development and model comparisons

Development: 1,280 events / 640 parameter groups / 11,520 windows. Three grouped folds; each fold's training side has separate fit, calibration and threshold groups. Group identities include the full generating configuration, and independent noise repetitions never cross partitions. Median imputation/missingness indicators are fitted on training only. Labels remain outside feature building.

Final bounded search: 20 attempts, 17 complete and 3 pruned, plus fixed feature ablations and an original-parameter refit. The rejected pre-cadence draft used another 20 development-only attempts. No test-driven optimization cycles were run. Models: multiclass XGBoost, Random Forest, Critical-versus-rest then Warning-versus-Normal XGBoost, and the fixed physical rule. Search includes inverse-frequency/cost weighting, none/sigmoid/isotonic calibration, displacement/movement/full feature groups and causal EWMA smoothing.

Grouped development selection estimates (not independent acceptance):

{table(cv.to_dict('records'))}

Reserved noisy development validation: 512 events / 256 groups, same development shape families with new parameters and 1.5× noise. Selected model: **98.48% Critical recall / 0.026% Normal FPR**. Its group interval gates passed before test generation. Fold/model/ablation records are saved; the validation set did not select a different candidate after the model was chosen by CV.

Independent test, same target and samples for every row:

{table(test['metrics'])}

The fixed displacement rule is strongest overall on this revised, directly observable task. The refitted original XGBoost settings also have substantially better macro F1 than the selected two-stage model. The candidate trades Warning sensitivity for an especially low Normal FPR; its saved Warning probability threshold is 1.0. **There is no evidence that the complex model is necessary for this local threshold task.** The frozen selection was not changed using this test comparison. Existing deployed artifacts were not replaced.

The exact original frozen artifact was separately evaluated on the same new observations: {100*old_legacy['critical_recall']:.2f}% Critical recall against legacy scenario labels. The selected v3 model scored {100*new_legacy['critical_recall']:.2f}% against those retained labels. Neither meets the old 89% target. Cross-task scores against the new severity labels are included only as diagnostics in the JSON/CSV, not as a fair same-task improvement claim.

Normal FPR means any WARNING or CRITICAL prediction on true NORMAL windows. Macro AP is average precision across three one-versus-rest tasks, not trapezoidal PR integration. Brier is summed three-class squared probability error; ECE uses 15 confidence bins. Anomaly recall is not used as Critical recall.

## Alerts, latency and limits

The local detector found **138/142 Critical events (97.18%)**, missing four. For persistence 1/2/3 windows, median delays after the true 35 mm crossing were 3.17/4.83/6.50 hours among detected events. There were zero false Critical episodes in 798 entirely Normal events. These are separately reported local detector results; they do not bypass the regional AlertEngine's unavailable-neighbor gate.

The long inherited 60-sample window has about 9.83 hours of warmup at ten-minute cadence; outputs then arrive every ten samples. This makes detection late despite high classification recall. Delay is relative to a synthetic severity crossing, **not collapse lead time or exact collapse prediction**. Regional multi-node confirmation/recovery validation remains unavailable and unchanged.

Median warm single-feature-row inference was {latency['median_ms']:.3f} ms (p95 {latency['p95_ms']:.3f} ms), excluding raw feature construction and communication. The selected EWMA uses earlier windows when scoring a complete series; one-row timing has no prior smoothing context. This is workstation timing, not a physical sensor/edge benchmark.

The simulator has shared mathematical assumptions, sensor noise is configured rather than field-estimated, and 15/35 mm is a borrowed prototype policy. Test shape families were sigmoid, smoothstep and two-stage, wholly excluded from development/validation generation (exponential, power and ramp). Other domains/sites and real sensor failures remain unvalidated. Correcting the data and target semantics accounts for much of the performance change.

## Frozen artifacts, tests and reproduction

- Model: `models/generalization_v3/candidates.joblib` (linked from the run directory); selected two-stage model, competitors, trained imputers, isotonic calibrators, thresholds and smoothing settings.
- `selected_config.yaml`, `protocol.json`, `freeze.json`: explicit target policy, parameters, features and code/config/artifact digests. Evaluation refuses changed frozen sources.
- `independent_lock.json`: claimed exclusively before generation, one evaluation recorded, corpus/result hashes. A second invocation was rejected before opening test data. Prior model and consumed-test lock hashes remain unchanged.
- `{test_status}`. Physical coupling, drift units, finite-endpoint velocity, label-blindness, causal truncation/smoothing, class policy boundaries, deterministic noise repetitions and conflicting legacy labels have regression coverage.
- Raw-observation inference smoke: nine v3 predictions from a development event without labels. `validation.png` was visually inspected.

Run from repository root with the existing Python 3.12 environment:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -o addopts='' -q
.venv/bin/python scripts/report_generalization_v3.py
```

The executed modeling/evaluation commands were:

```sh
.venv/bin/python scripts/run_generalization_v3.py develop
.venv/bin/python scripts/run_generalization_v3.py evaluate
```

They now refuse to overwrite/reuse the frozen run. To reproduce in a separate directory:

```sh
.venv/bin/python scripts/run_generalization_v3.py develop --out reports/generalization_v3_reproduction
.venv/bin/python scripts/run_generalization_v3.py evaluate --out reports/generalization_v3_reproduction
.venv/bin/python scripts/report_generalization_v3.py --out reports/generalization_v3_reproduction
```

That reproduces the **same already-consumed test specification**, not another independent acceptance attempt; it must not drive tuning. A future development change needs a separately predeclared fresh evaluation and its own frozen checkout/protocol. Keep the current frozen sources/configurations to reproduce or use this artifact.

Label-free inference on a resampled ten-minute observation stream:

```sh
.venv/bin/python scripts/run_generalization_v3.py predict --input reports/generalization_v3/development_observations.parquet --predictions /tmp/bhurakshak-v3-demo.csv
```

Inputs require event/node/time keys plus displacement (mm), tilt_x/y (degrees), dimensionless strain, vibration_rms/peak and packet_loss. V3 rates use channel units/hour and acceleration uses channel units/hour squared. Missing values use training-fitted preprocessing; retain full event history for causal smoothing. Do not use locked test observations for ad hoc inference.

Important changed files: `src/simulator/coupled_v3.py`; `src/features/windowing.py`; `src/features/causal_v3.py`; `src/risk/severity_v3.py`; `configs/generalization_v3.yaml`; `configs/feature_schema_v3.yaml`; `scripts/run_generalization_v3.py`; `scripts/report_generalization_v3.py`; `tests/test_coupled_v3.py`; `.gitignore`; the new model and report directories. Old data, labels, models and prior results remain intact.

Next work is engineering validation, not a claim of completed mine safety: test the fixed rule and candidate against independent one-trapdoor measurements; validate site-appropriate severity boundaries; reduce causal warmup/alert delay under a new protocol; repair weak Warning recall on development data; then use another untouched test. The original scenario-label objective remains unresolved and cannot be claimed from this experiment.
'''
    (out/'final_report.md').write_text(report)
    combined=pd.concat([cv,pd.DataFrame(val['metrics']),pd.DataFrame(test['metrics'])],ignore_index=True)
    combined.to_csv(out/'model_comparison.csv',index=False)
    runtime=dict(python=sys.version,platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','xgboost','optuna','joblib','pyarrow']})
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    print(out/'final_report.md')

if __name__=='__main__':main()
