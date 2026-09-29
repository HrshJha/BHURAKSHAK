#!/usr/bin/env python3
"""One-use, fresh-seed synthetic evaluation after candidate freeze.

Existing heldout_locked files are never opened. A prior evaluation lock,
including a failed attempt, refuses reruns. Reproduction may inspect saved
aggregate outputs but must not feed them back into optimization.
"""
from __future__ import annotations
import json
import argparse
import sys
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
import yaml
from scripts.optimize_bhurakshak import OUT,sha,dump,metrics,if_metrics,alerts
from scripts.final_eval import _aligned_prob, _calibrate, _threshold_predictions
from src.features.build_feature_store import build_feature_store
from src.simulator.dataset_builder import build_dataset


def claim(path, lock):
    # Exclusive creation also protects against concurrent double evaluation.
    with Path(path).open('x') as f:json.dump(lock,f,indent=2)


def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='reports/optimization')
    ap.add_argument('--test-dir',default='data/optimization_independent_v1');args=ap.parse_args()
    OUT=(ROOT/args.output_dir).resolve()
    freeze=json.loads((OUT/'freeze.json').read_text())
    artifact=ROOT/freeze.get('model_path','models/optimization/candidate.joblib')
    config=OUT/'best_development_config.yaml'
    if sha(artifact)!=freeze['model_sha256'] or sha(config)!=freeze['config_sha256']:
        raise SystemExit('frozen artifact/config digest mismatch')
    cfg=yaml.safe_load(config.read_text())['config']['independent_test']
    baseline_paths=[ROOT/'models/tuned/risk_xgboost_tuned.joblib',ROOT/'models/tuned/iforest_tuned.joblib']
    # Snapshot original comparators and feature-builder sources before generation.
    lock=dict(status='claimed_before_generation',evals_run=1,model_sha256=freeze['model_sha256'],config_sha256=freeze['config_sha256'],
         original_model_hashes={str(p.relative_to(ROOT)):sha(p) for p in baseline_paths},test_spec=cfg,
         source_hashes={str(p.relative_to(ROOT)):sha(p) for folder in ['src/features','src/simulator'] for p in (ROOT/folder).glob('*.py')},
         old_lock_sha256=sha(ROOT/'reports/test_lock.json'))
    lockpath=OUT/'independent_test_lock.json'
    claim(lockpath,lock)
    target=(ROOT/args.test_dir).resolve()
    if target.exists():raise SystemExit('fresh test destination exists; refusing to replace it')
    result=build_dataset(target,sequences_per_scenario=cfg['sequences_per_scenario'],seed=cfg['seed'],dataset_version=cfg['dataset_version'],scenario_names=tuple(cfg['scenarios']))
    lock.update(status='generated',corpus_hashes=result['artifact_sha256']);dump(lockpath,lock)
    raw=pd.read_csv(target/'synthetic_nodes.csv');coords=raw[['node_id','x','y']].drop_duplicates('node_id')
    frame,report=build_feature_store(raw,coords,center_mode='detected')
    frame['scenario_family']=frame.event_id.str.rsplit('_',n=2).str[0]
    models=joblib.load(artifact);records=[];scenario_rows=[]
    for name,key in [('original_config_refit','baseline'),('optimized_risk','risk')]:
        model=models[key];prob=model.predict_proba(frame);pred=model.predict(frame)
        records.append(dict(experiment=name,scope='independent_synthetic_test',**metrics(frame,prob,pred)))
        for family,g in frame.groupby('scenario_family'):
            ix=g.index.to_numpy();critical=g.risk_label=='CRITICAL';normal=g.risk_label=='NORMAL'
            scenario_rows.append(dict(model=name,scenario=family,rows=len(g),critical_recall=float(np.mean(pred[ix][critical]=='CRITICAL')) if critical.any() else None,
                    normal_fpr=float(np.mean(pred[ix][normal]!='NORMAL')) if normal.any() else None))
    for name,key in [('original_if_refit','baseline_anomaly'),('optimized_if','anomaly')]:
        model=models[key];score=-model['model'].score_samples(frame[model['features']].to_numpy(float))
        records.append(dict(experiment=name,scope='independent_synthetic_test',**if_metrics(frame,score,model['threshold'])))
    # Exact old frozen artifacts, including their old threshold semantics.
    old=joblib.load(baseline_paths[0]);prob=_calibrate(_aligned_prob(old['model'],frame[old['features']].to_numpy(float)),old['calibration'])
    pred=_threshold_predictions(prob,old['thresholds'])
    records.append(dict(experiment='original_frozen_artifact',scope='independent_synthetic_test',**metrics(frame,prob,pred)))
    oldif=joblib.load(baseline_paths[1]);score=-oldif['model'].score_samples(frame[oldif['features']].to_numpy(float))
    records.append(dict(experiment='original_frozen_if_artifact',scope='independent_synthetic_test',**if_metrics(frame,score,oldif['score_threshold'])))
    a=models['alert'];pr=models['risk'].predict_proba(frame)
    alert_results=dict(original_frozen=alerts(frame,prob),optimized=alerts(frame,pr,a['watch_threshold'],a['persistence_windows']))
    dump(OUT/'independent_metrics.json',dict(metrics=records,alerts=alert_results,scenario_results=scenario_rows,feature_windows=len(frame),
       events=frame.event_id.nunique(),scope='synthetic same-generator, unseen scenario families; not field validation',
       pass_target=next(r for r in records if r['experiment']=='optimized_risk')['critical_recall']>=.89 and next(r for r in records if r['experiment']=='optimized_risk')['normal_fpr']<=.05))
    comparison=pd.read_csv(OUT/'model_comparison.csv');pd.concat([comparison,pd.DataFrame(records)],ignore_index=True).to_csv(OUT/'model_comparison.csv',index=False)
    lock.update(status='completed',test_feature_rows=len(frame),result_sha256=sha(OUT/'independent_metrics.json'));dump(lockpath,lock)
    assert sha(ROOT/'reports/test_lock.json')==lock['old_lock_sha256']
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
