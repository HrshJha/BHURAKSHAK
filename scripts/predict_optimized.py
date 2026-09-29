#!/usr/bin/env python3
"""Apply the frozen experimental candidate to precomputed label-blind features.

This is an explicit experimental entry point, not a replacement of deployed
models. No training, calibration fitting or threshold selection is performed.
"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import pandas as pd
from scripts.optimize_bhurakshak import sha
from src.risk.optimized import CLASSES


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--run-dir',default='reports/optimization')
    args=ap.parse_args();run=ROOT/args.run_dir
    freeze=json.loads((run/'freeze.json').read_text())
    artifact=ROOT/freeze.get('model_path','models/optimization/candidate.joblib')
    if sha(artifact)!=freeze['model_sha256']:raise SystemExit('candidate digest mismatch')
    if sha(run/'best_development_config.yaml')!=freeze['config_sha256']:raise SystemExit('configuration digest mismatch')
    path=Path(args.features);target=Path(args.output)
    if target.exists():raise SystemExit('output exists; refusing to overwrite')
    frame=pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)
    bundle=joblib.load(artifact);risk=bundle['risk'];anomaly=bundle['anomaly']
    prob=risk.predict_proba(frame)
    result=frame[[c for c in ['event_id','node_id','window_index','window_timestamp'] if c in frame]].copy()
    for i,label in enumerate(CLASSES):result[f'p_{label}']=prob[:,i]
    result['risk_prediction']=risk.predict(frame)
    result['anomaly_score']=-anomaly['model'].score_samples(frame[anomaly['features']].to_numpy(float))
    result['anomaly_flag']=result.anomaly_score>=anomaly['threshold']
    result['model_status']='experimental_acceptance_target_not_met'
    result.to_csv(target,index=False)
    print(f'wrote {len(result)} experimental predictions to {target}')

if __name__=='__main__':main()
