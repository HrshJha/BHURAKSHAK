#!/usr/bin/env python3
"""Physically coupled v3 development, frozen independent evaluation and inference.

Never reads prior consumed test datasets. Target is explicitly the existing
15/35 mm local-severity policy; old scenario labels are reported separately.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yaml
from scipy.special import ndtr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import scripts.optimize_bhurakshak as shared
from scripts.freeze_models import _calibrate
from src.simulator.coupled_v3 import generate_corpus
from src.features.causal_v3 import build_features,attach_targets,schema
from src.risk.severity_v3 import SeverityBundle,TwoStageClassifier
from src.risk.optimized import CLASSES,threshold_predictions


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):
    def convert(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        return str(x)
    Path(path).write_text(json.dumps(obj,indent=2,default=convert)+'\n')

def source_hashes():
    names=['configs/generalization_v3.yaml','configs/feature_schema_v3.yaml','src/simulator/coupled_v3.py','src/features/causal_v3.py',
           'src/features/windowing.py','src/risk/severity_v3.py','src/risk/optimized.py','scripts/run_generalization_v3.py',
           'scripts/optimize_bhurakshak.py','scripts/freeze_models.py','configs/physics.yaml','src/simulator/faults.py']
    return {n:sha(ROOT/n) for n in names}

def generate(cfg,section,out):
    raw,truth,parameters=generate_corpus(cfg,section)
    features=build_features(raw);frame=attach_targets(features,truth)
    assert len(frame)==len(features) and frame.risk_label.notna().all()
    if not set(schema()['features']).isdisjoint(set(schema()['excluded'])):raise ValueError('feature provenance violation')
    raw.to_parquet(out/f'{section}_observations.parquet',index=False)
    truth.to_parquet(out/f'{section}_truth.parquet',index=False)
    frame.to_parquet(out/f'{section}_features.parquet',index=False)
    parameters.to_csv(out/f'{section}_parameters.csv',index=False)
    return frame,raw,truth,parameters


def feature_choices(frame):
    full=schema()['features'];movement=[f for f in full if not f.startswith('vibration')]
    disp=[f for f in full if f.startswith('displacement') or f in ['channel_missing_ratio','packet_loss']]
    return dict(full=full,movement=movement,displacement=disp)


def make_model(p,seed):
    kind=p['model']
    if kind=='xgboost':
        return XGBClassifier(n_estimators=p['n_estimators'],max_depth=p['max_depth'],learning_rate=p['learning_rate'],min_child_weight=p.get('min_child_weight',2.),
           reg_lambda=p.get('reg_lambda',2.),subsample=p.get('subsample',.9),colsample_bytree=p.get('colsample_bytree',.9),objective='multi:softprob',num_class=3,
           eval_metric='mlogloss',tree_method='hist',n_jobs=1,random_state=seed,**{k:p[k] for k in ['gamma','reg_alpha','colsample_bylevel','max_delta_step','max_bin'] if k in p})
    if kind=='random_forest':
        return RandomForestClassifier(n_estimators=p['n_estimators'],max_depth=p['max_depth'],min_samples_leaf=p.get('min_samples_leaf',2),max_features=.9,n_jobs=1,random_state=seed)
    if kind=='two_stage':return TwoStageClassifier(p['n_estimators'],p['max_depth'],p['learning_rate'],seed)
    raise ValueError(kind)


def fit(parts,p,features,cfg):
    train,cal,decision=parts
    y=train.risk_label.map({c:i for i,c in enumerate(CLASSES)}).to_numpy()
    counts=np.bincount(y,minlength=3);weights=(len(y)/(3*counts))[y]**p.get('weight_power',0.)
    weights[y==2]*=p.get('critical_weight',1.)
    pipe=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('estimator',make_model(p,cfg['seed']))])
    pipe.fit(train[features].to_numpy(float),y,estimator__sample_weight=weights)
    method=p.get('calibration','none')
    calibration=None if method=='none' else _calibrate(pipe.predict_proba(cal[features].to_numpy(float)),cal.risk_label.to_numpy(),method)
    model=SeverityBundle(pipe,features,calibration,{},p.get('alpha',1.))
    model.thresholds=shared.select_thresholds(decision,model.predict_proba(decision),cfg['selection_normal_fpr_cap'])
    return model


def metric(frame,prob,pred):return shared.metrics(frame,prob,pred)

def quality(m):
    return m['critical_recall']+.2*m['macro_f1']+.1*m['macro_pr_auc']-10*max(0,m['normal_fpr']-.025)


def cv_evaluate(plan,p,features,cfg,trial=None):
    rows=[];oof=[]
    for i,(parts,val) in enumerate(plan):
        model=fit(parts,p,features,cfg);prob=model.predict_proba(val);pred=threshold_predictions(prob,model.thresholds)
        m=metric(val,prob,pred);m['fold']=i;rows.append(m)
        chunk=pd.DataFrame(dict(index=val.index,p0=prob[:,0],p1=prob[:,1],p2=prob[:,2],prediction=pred));oof.append(chunk)
        if trial:
            trial.set_user_attr('folds',rows);trial.report(quality(shared.mean_metrics(rows)),i)
            if trial.should_prune():raise optuna.TrialPruned()
    return shared.mean_metrics(rows),rows,pd.concat(oof).sort_values('index')


def suggest(t):
    kind=t.suggest_categorical('model',['xgboost','random_forest','two_stage'])
    p=dict(model=kind,n_estimators=t.suggest_int('n_estimators',150,450,step=100),max_depth=t.suggest_int('max_depth',3,8),
           learning_rate=t.suggest_float('learning_rate',.03,.16,log=True),critical_weight=t.suggest_float('critical_weight',1,3),
           weight_power=t.suggest_categorical('weight_power',[0.,.5,1.]),calibration=t.suggest_categorical('calibration',['none','sigmoid','isotonic']),
           feature_set=t.suggest_categorical('feature_set',['displacement','movement','full']),alpha=t.suggest_categorical('alpha',[1.,.7]))
    if kind=='xgboost':p.update(min_child_weight=t.suggest_float('min_child_weight',1,12),reg_lambda=t.suggest_float('reg_lambda',.5,10,log=True),colsample_bytree=t.suggest_float('colsample_bytree',.7,1.))
    if kind=='random_forest':p['min_samples_leaf']=t.suggest_int('min_samples_leaf',1,6)
    return p


def uncertainty(frame,pred,reps=500):
    """Resample whole generating groups, preserving event windows/noise replicas."""
    y=frame.risk_label.to_numpy();work=pd.DataFrame(dict(group=frame.generation_parameter_id,critical=(y=='CRITICAL').astype(int),
         critical_hit=((y=='CRITICAL')&(pred=='CRITICAL')).astype(int),normal=(y=='NORMAL').astype(int),false_alarm=((y=='NORMAL')&(pred!='NORMAL')).astype(int)))
    counts=work.groupby('group')[['critical','critical_hit','normal','false_alarm']].sum().to_numpy();rng=np.random.default_rng(442)
    values=[]
    for _ in range(reps):
        a=counts[rng.integers(0,len(counts),len(counts))].sum(0)
        if a[0]>0 and a[2]>0:values.append([a[1]/a[0],a[3]/a[2]])
    a=np.asarray(values)
    return dict(unit='generation_parameter_id (whole events and repeat noise realizations)',groups=len(counts),replicates=len(a),
        critical_recall_95pct=np.quantile(a[:,0],[.025,.975]).tolist(),normal_fpr_95pct=np.quantile(a[:,1],[.025,.975]).tolist())


def physics_predictions(frame,cfg):
    # Fixed pre-existing physical policy; sensor observations only, no latent targets.
    value=frame.displacement_endpoint.to_numpy(float);value=np.nan_to_num(value,nan=0.)
    warning=cfg['label_policy']['warning_mm'];critical=cfg['label_policy']['critical_mm']
    pred=np.where(value>critical,'CRITICAL',np.where(value>warning,'WARNING','NORMAL'))
    sigma=1.;a=ndtr((warning-value)/sigma);b=ndtr((critical-value)/sigma)
    prob=np.column_stack([a,b-a,1-b]);return prob,pred


def events(frame,pred,truth,persistence=1):
    work=frame[['event_id','window_timestamp','risk_label']].copy();work['pred']=pred
    critical_events=detected=0;delays=[];normal_events=false_events=episodes=0
    for eid,g in work.groupby('event_id',sort=False):
        g=g.sort_values('window_timestamp');reference=truth[truth.event_id==eid]
        critical_times=reference.loc[reference.risk_label=='CRITICAL','timestamp']
        active=[];streak=0
        for value in g.pred:
            streak=streak+1 if value=='CRITICAL' else 0;active.append(streak>=persistence)
        flag=np.array(active)
        if len(critical_times):
            critical_events+=1;onset=float(critical_times.min());times=g.loc[flag & (g.window_timestamp>=onset),'window_timestamp']
            if len(times):detected+=1;delays.append(float(times.min()-onset))
        if (reference.risk_label=='NORMAL').all():
            normal_events+=1;false_events+=int(flag.any());episodes+=int(np.sum(flag & ~np.r_[False,flag[:-1]]))
    return dict(persistence_windows=persistence,critical_events=critical_events,detected_critical_events=detected,missed_critical_events=critical_events-detected,
        critical_event_recall=detected/critical_events if critical_events else None,normal_events=normal_events,false_critical_event_fraction=false_events/max(normal_events,1),
        false_critical_episodes=episodes,median_detection_delay_hours=float(np.median(delays)) if delays else None,
        p90_detection_delay_hours=float(np.quantile(delays,.9)) if delays else None,
        meaning='local severity detector; delay from true 35 mm crossing among detected events, not collapse lead time; regional neighbor gate unchanged')


def score_models(frame,models,truth,cfg,scope):
    records=[];uncertainties={};event_results={};scenarios=[]
    for name,model in models.items():
        if name=='physics_rule':prob,pred=physics_predictions(frame,cfg)
        else:prob=model.predict_proba(frame);pred=threshold_predictions(prob,model.thresholds)
        records.append(dict(model=name,scope=scope,target='local_severity_v1',**metric(frame,prob,pred)))
        uncertainties[name]=uncertainty(frame,pred,cfg['bootstrap_replicates'])
        event_results[name]=[events(frame,pred,truth,p) for p in [1,2,3]]
        for family,idx in frame.groupby('scenario_family').indices.items():
            y=frame.iloc[idx].risk_label.to_numpy();pp=pred[idx]
            scenarios.append(dict(model=name,scenario=family,rows=len(idx),critical_rows=int(np.sum(y=='CRITICAL')),normal_rows=int(np.sum(y=='NORMAL')),
                critical_recall=float(np.mean(pp[y=='CRITICAL']=='CRITICAL')) if (y=='CRITICAL').any() else None,
                normal_fpr=float(np.mean(pp[y=='NORMAL']!='NORMAL')) if (y=='NORMAL').any() else None))
        if name=='selected':
            alternate=frame.copy();alternate['risk_label']=alternate.legacy_risk_label
            records.append(dict(model='selected_vs_legacy_scenario_labels',scope=scope,target='legacy_scenario_taxonomy_different_task',**metric(alternate,prob,pred)))
    return dict(metrics=records,uncertainty=uncertainties,events=event_results,by_scenario=scenarios)


def develop(cfg,out):
    if (out/'freeze.json').exists():raise SystemExit('v3 candidate already frozen; use a new output directory')
    protocol=dict(config=cfg,sources=source_hashes(),prior_models={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'models/tuned/risk_xgboost_tuned.joblib',ROOT/'models/optimization/candidate.joblib']},
       prior_locks={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'reports/test_lock.json',ROOT/'reports/optimization/independent_test_lock.json']})
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise SystemExit('protocol changed; refusing to reuse development/study artifacts')
    dump(out/'protocol.json',protocol)
    print('Generating physically checked development corpus',flush=True)
    frame,raw,truth,parameters=generate(cfg,'development',out)
    dump(out/'development_audit.json',dict(rows=len(frame),events=frame.event_id.nunique(),parameter_groups=frame.generation_parameter_id.nunique(),
        class_counts=frame.risk_label.value_counts().to_dict(),legacy_class_counts=frame.legacy_risk_label.value_counts().to_dict(),
        scenario_critical_but_locally_normal=float(np.mean((frame.legacy_risk_label=='CRITICAL')&(frame.risk_label=='NORMAL'))),
        missing_features=frame[schema()['features']].isna().sum().to_dict()))
    sets=feature_choices(frame);shared.SEED=cfg['seed'];plan=shared.folds(frame,cfg['folds'])
    pd.DataFrame([dict(event_id=e,fold=i) for i,(_,v) in enumerate(plan) for e in v.event_id.unique()]).to_csv(out/'fold_assignment.csv',index=False)
    study=optuna.create_study(study_name='v3',direction='maximize',storage=f'sqlite:///{out/"study.sqlite3"}',load_if_exists=True,
       sampler=optuna.samplers.TPESampler(seed=cfg['seed']),pruner=optuna.pruners.MedianPruner(n_startup_trials=6,n_warmup_steps=1))
    if not study.trials:
        for kind in ['xgboost','random_forest','two_stage']:
            study.enqueue_trial(dict(model=kind,n_estimators=250,max_depth=5,learning_rate=.08,critical_weight=1.5,weight_power=0.,calibration='sigmoid',feature_set='movement',alpha=1.))
    def objective(t):
        p=suggest(t);m,rows,_=cv_evaluate(plan,p,sets[p['feature_set']],cfg,t);t.set_user_attr('metrics',m);return quality(m)
    finished=sum(t.state.is_finished() for t in study.trials)
    study.optimize(objective,n_trials=max(0,cfg['trials']-finished),timeout=1800)
    dump(out/'study.json',[dict(number=t.number,state=t.state.name,value=t.value,params=t.params,attrs=t.user_attrs) for t in study.trials])
    records=[];configs={};oofs={}
    for kind in ['xgboost','random_forest','two_stage']:
        candidates=[t for t in study.trials if t.state.name=='COMPLETE' and t.params['model']==kind]
        if not candidates:continue
        t=max(candidates,key=lambda t:t.value);p=t.params
        m,rows,oof=cv_evaluate(plan,p,sets[p['feature_set']],cfg);records.append(dict(model=kind,scope='grouped_cv_selection',**m));configs[kind]=p;oofs[kind]=oof
        dump(out/f'{kind}_folds.json',rows);oof.to_csv(out/f'{kind}_oof.csv',index=False)
    # Same-model feature ablation using the CV winner's settings, selected only on training CV.
    current=max(records,key=quality);best=dict(configs[current['model']]);winner=current['model']
    for feature_set in sets:
        if feature_set==best['feature_set']:continue
        p=dict(best,feature_set=feature_set);name='ablation_'+feature_set
        m,rows,oof=cv_evaluate(plan,p,sets[feature_set],cfg);configs[name]=p;oofs[name]=oof;records.append(dict(model=name,scope='grouped_cv_selection',**m));dump(out/f'{name}_folds.json',rows)
    # Refit original XGBoost hyperparameters on the same v3 target/features/partitions.
    old=yaml.safe_load((ROOT/'configs/model_params.yaml').read_text())['risk_classifier']['params']
    baseline={k:v for k,v in old.items() if k not in ['class_weighting','critical_threshold','warning_threshold']}
    baseline.update(model='xgboost',n_estimators=2000,critical_weight=1.,weight_power=1.,calibration='isotonic',feature_set='movement',alpha=1.)
    configs['original_hyperparameters_refit']=baseline
    m,rows,oof=cv_evaluate(plan,baseline,sets['movement'],cfg);records.append(dict(model='original_hyperparameters_refit',scope='grouped_cv_selection',**m));oofs['original_hyperparameters_refit']=oof;dump(out/'original_refit_folds.json',rows)
    # Physics rule has no fitted parameters and participates in the same CV-row comparison.
    pp,pr=physics_predictions(frame,cfg);m=metric(frame,pp,pr);records.append(dict(model='physics_rule',scope='grouped_cv_selection',**m))
    eligible=[r for r in records if r['critical_recall']>=cfg['development_margin']['critical_recall'] and r['normal_fpr']<=cfg['development_margin']['normal_fpr']]
    if not eligible:
        dump(out/'development_failure.json',dict(reason='no candidate meets development safety margin',records=records));raise SystemExit('Development acceptance not met; no test generated')
    # Keep a learned candidate for explicit model/inference evaluation; compare physics baseline separately.
    learned=[r for r in eligible if r['model']!='physics_rule']
    if not learned:raise SystemExit('Only the fixed physics baseline passes; learned candidate not frozen')
    winner=max(learned,key=quality)['model'];best=configs[winner]
    pd.DataFrame(records).to_csv(out/'development_comparison.csv',index=False)
    trials=[dict(experiment=f'trial_{t.number}',state=t.state.name,value=t.value,parameters=json.dumps(t.params),**t.user_attrs.get('metrics',{})) for t in study.trials]
    pd.DataFrame(trials).to_csv(out/'experiment_log.csv',index=False)
    parts=shared.inner_parts(frame);trained={}
    for name,p in configs.items():
        if name.startswith('ablation') and name!=winner:continue
        trained[name]=fit(parts,p,sets[p['feature_set']],cfg)
    trained['selected']=trained[winner];trained['physics_rule']=None
    validation,vraw,vtruth,vparams=generate(cfg,'validation',out)
    assert not set(frame.generation_parameter_id)&set(validation.generation_parameter_id)
    result=score_models(validation,trained,vtruth,cfg,'reserved_noisy_development_validation');dump(out/'validation.json',result)
    selected=next(r for r in result['metrics'] if r['model']=='selected');ci=result['uncertainty']['selected']
    margin=selected['critical_recall']>=cfg['development_margin']['critical_recall'] and selected['normal_fpr']<=cfg['development_margin']['normal_fpr']
    confidence=ci['critical_recall_95pct'][0]>=cfg['acceptance']['critical_recall'] and ci['normal_fpr_95pct'][1]<=cfg['acceptance']['normal_fpr']
    if not (margin and confidence):
        dump(out/'validation_failure.json',dict(selected=selected,uncertainty=ci,margin=margin,confidence=confidence));joblib.dump(trained,out/'unaccepted_candidates.joblib');raise SystemExit('Validation margin/uncertainty gate failed; no independent test generated')
    joblib.dump(trained,out/'candidates.joblib')
    saved=dict(selected_name=winner,params=best,thresholds=trained['selected'].thresholds,features=trained['selected'].features,config=cfg,
        feature_version='v3',target='local_displacement_severity_v1',legacy_target_not_claimed=True)
    (out/'selected_config.yaml').write_text(yaml.safe_dump(saved,sort_keys=False))
    frozen=dict(model_sha256=sha(out/'candidates.joblib'),config_sha256=sha(out/'selected_config.yaml'),sources=source_hashes(),protocol_sha256=sha(out/'protocol.json'),
        selected_name=winner,development_margin_met=True,group_ci_gate_met=True,test_generated=False,development_group_ids=parameters.generation_parameter_id.tolist(),validation_group_ids=vparams.generation_parameter_id.tolist())
    dump(out/'freeze.json',frozen)
    # Diagnostics only; no selection after validation.
    model=trained['selected'];prob=model.predict_proba(validation);pred=model.predict(validation)
    pd.DataFrame(dict(event_id=validation.event_id,truth=validation.risk_label,prediction=pred,p_critical=prob[:,2])).to_csv(out/'validation_predictions.csv',index=False)
    figure,axes=plt.subplots(1,2,figsize=(10,4));cm=np.asarray(selected['confusion_matrix']);axes[0].imshow(cm,cmap='Blues')
    for i in range(3):
        for j in range(3):axes[0].text(j,i,str(cm[i,j]),ha='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    axes[0].set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,title='v3 validation confusion',xlabel='Predicted',ylabel='True local severity')
    from sklearn.calibration import calibration_curve
    for i,c in enumerate(CLASSES):
        a,b=calibration_curve(validation.risk_label.to_numpy()==c,prob[:,i],n_bins=8,strategy='quantile');axes[1].plot(b,a,'o-',label=c)
    axes[1].plot([0,1],[0,1],'k--');axes[1].legend();axes[1].set(xlabel='Probability',ylabel='Observed fraction',title='Calibration, noisy validation')
    figure.tight_layout();figure.savefig(out/'validation.png',dpi=150);plt.close(figure)
    start=time.perf_counter();model.predict(validation.iloc[:1]);times=[]
    for _ in range(100):
        start=time.perf_counter();model.predict(validation.iloc[:1]);times.append((time.perf_counter()-start)*1000)
    dump(out/'latency.json',dict(median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),scope='warm one-row feature inference, excludes raw feature extraction; smoothing has no earlier context in one-row call'))
    print(json.dumps(dict(selected=selected,confidence_interval=ci,winner=winner),indent=2),flush=True)


def verify_freeze(out):
    frozen=json.loads((out/'freeze.json').read_text())
    if sha(out/'candidates.joblib')!=frozen['model_sha256'] or sha(out/'selected_config.yaml')!=frozen['config_sha256']:raise RuntimeError('frozen artifact/config changed')
    if source_hashes()!=frozen['sources']:raise RuntimeError('frozen code/config changed; refuse independent evaluation')
    return frozen


def evaluate(cfg,out):
    frozen=verify_freeze(out);lock=out/'independent_lock.json'
    with lock.open('x') as f:json.dump(dict(status='claimed_before_generation',evals_run=1,freeze_sha256=sha(out/'freeze.json'),test_spec=cfg['independent_test']),f,indent=2)
    frame,raw,truth,params=generate(cfg,'independent_test',out)
    assert not set(params.generation_parameter_id)&set(frozen['development_group_ids']+frozen['validation_group_ids'])
    assert set(frame.shape_family).isdisjoint(set(cfg['development']['shapes'])|set(cfg['validation']['shapes']))
    models=joblib.load(out/'candidates.joblib');result=score_models(frame,models,truth,cfg,'independent_unseen_shape_test')
    # Original exact artifact comparison is across target semantics and is labelled accordingly.
    from src.features.build_feature_store import build_feature_store
    from scripts.final_eval import _aligned_prob,_calibrate as calibrate_original,_threshold_predictions
    oldfeatures,_=build_feature_store(raw,raw[['node_id','x','y']].drop_duplicates('node_id'))
    oldfeatures=frame[['event_id','node_id','window_index']].merge(oldfeatures,on=['event_id','node_id','window_index'],validate='one_to_one')
    old=joblib.load(ROOT/'models/tuned/risk_xgboost_tuned.joblib');prob=calibrate_original(_aligned_prob(old['model'],oldfeatures[old['features']].to_numpy(float)),old['calibration']);pred=_threshold_predictions(prob,old['thresholds'])
    result['metrics'].append(dict(model='original_frozen_cross_task_diagnostic',scope='independent_unseen_shape_test',target='local_severity_v1_not_original_training_target',**metric(frame,prob,pred)))
    legacy=frame.copy();legacy['risk_label']=legacy.legacy_risk_label
    result['metrics'].append(dict(model='original_frozen_legacy_taxonomy',scope='independent_unseen_shape_test',target='legacy_scenario_taxonomy',**metric(legacy,prob,pred)))
    selected=next(r for r in result['metrics'] if r['model']=='selected');ci=result['uncertainty']['selected']
    result['acceptance']=dict(target_policy='local_displacement_severity_v1',legacy_target_not_claimed=True,
       point_estimates_pass=selected['critical_recall']>=.89 and selected['normal_fpr']<=.05,
       group_95pct_bounds_pass=ci['critical_recall_95pct'][0]>=.89 and ci['normal_fpr_95pct'][1]<=.05,
       test_events=int(frame.event_id.nunique()),parameter_groups=int(frame.generation_parameter_id.nunique()),windows=len(frame),unseen_shapes=sorted(frame.shape_family.unique()),
       qualification='synthetic prototype-scale local severity classification; not validation of original scenario-label target, field safety or collapse prediction')
    dump(out/'independent_results.json',result)
    record=json.loads(lock.read_text());record.update(status='completed',result_sha256=sha(out/'independent_results.json'),
       corpus_hashes={name:sha(out/name) for name in ['independent_test_observations.parquet','independent_test_truth.parquet','independent_test_parameters.csv']})
    dump(lock,record)
    original=json.loads((out/'protocol.json').read_text())
    for path,digest in {**original['prior_models'],**original['prior_locks']}.items():assert sha(ROOT/path)==digest
    print(json.dumps(dict(selected=selected,uncertainty=ci,acceptance=result['acceptance']),indent=2),flush=True)


def predict(args,out):
    verify_freeze(out);model=joblib.load(out/'candidates.joblib')['selected'];p=Path(args.input)
    raw=pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)
    frame=build_features(raw);prob=model.predict_proba(frame);output=frame[['event_id','node_id','window_index','window_timestamp']].copy()
    for i,c in enumerate(CLASSES):output['p_'+c]=prob[:,i]
    output['local_severity']=model.predict(frame);output['target_policy']=model.target_policy
    path=Path(args.predictions)
    if path.exists():raise SystemExit('prediction output already exists')
    output.to_csv(path,index=False);print(f'wrote {len(output)} label-free v3 predictions')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['develop','evaluate','predict'])
    parser.add_argument('--out',default='reports/generalization_v3');parser.add_argument('--input');parser.add_argument('--predictions');args=parser.parse_args()
    out=(ROOT/args.out).resolve();out.mkdir(parents=True,exist_ok=True);cfg=yaml.safe_load((ROOT/'configs/generalization_v3.yaml').read_text())
    if args.action=='develop':develop(cfg,out)
    elif args.action=='evaluate':evaluate(cfg,out)
    else:
        if not args.input or not args.predictions:parser.error('predict requires --input and --predictions')
        predict(args,out)

if __name__=='__main__':main()
