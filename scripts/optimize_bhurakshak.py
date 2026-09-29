#!/usr/bin/env python3
"""Bounded, regime-grouped optimization. Never opens an existing test corpus.

Development selection is restricted to the original train partition. Within
an outer fold, fitting, calibration and threshold selection use disjoint
parameter groups. Original validation is evaluated only after selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from scripts.tune_models import development_frame
from scripts.freeze_models import _calibrate, _ece
from src.anomaly.isolation_forest import healthy_baseline_mask
from src.risk.optimized import CLASSES, OptimizedRiskModel, threshold_predictions

OUT = ROOT / 'reports/optimization'
SEED = 42
FPR_CAP = .05
FEATURE_CHOICES = ["displacement_health", "displacement_physics"]
CALIBRATION_CHOICES = ["none", "sigmoid", "isotonic"]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inner_parts(frame):
    """60/20/20 by generating regime, never by individual window."""
    groups = frame.generation_parameter_id
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.4, random_state=SEED).split(frame, groups=groups))
    fit, rest = frame.iloc[a], frame.iloc[b]
    c, d = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=SEED+1).split(rest, groups=rest.generation_parameter_id))
    parts = (fit, rest.iloc[c], rest.iloc[d])
    for part in parts:
        if set(part.risk_label) != set(CLASSES):
            raise ValueError('inner split lacks a risk class')
    for i in range(3):
        for j in range(i):
            assert not set(parts[i].generation_parameter_id) & set(parts[j].generation_parameter_id)
    return parts


def folds(frame, n=3):
    y = frame.risk_label.map({c:i for i,c in enumerate(CLASSES)})
    splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED)
    result = []
    for a,b in splitter.split(frame, y, frame.generation_parameter_id):
        tr, va = frame.iloc[a], frame.iloc[b]
        assert not set(tr.generation_parameter_id) & set(va.generation_parameter_id)
        result.append((inner_parts(tr), va))
    return result


def metrics(frame, prob, pred):
    y = frame.risk_label.to_numpy()
    truth = np.column_stack([y == c for c in CLASSES])
    normal = y == 'NORMAL'
    return dict(critical_recall=float(np.mean(pred[y=='CRITICAL']=='CRITICAL')),
                macro_pr_auc=float(np.mean([average_precision_score(truth[:,i],prob[:,i]) for i in range(3)])),
                macro_f1=float(f1_score(y,pred,labels=list(CLASSES),average='macro',zero_division=0)),
                normal_fpr=float(np.mean(pred[normal]!='NORMAL')),
                normal_critical_fpr=float(np.mean(pred[normal]=='CRITICAL')),
                brier=float(np.mean(np.sum((prob-truth)**2,axis=1))), ece=_ece(prob,y), rows=len(frame),
                confusion_matrix=confusion_matrix(y,pred,labels=list(CLASSES)).tolist())


def select_thresholds(frame, prob, cap=.05):
    # Joint NORMAL false alarm budget: first prioritize Critical recall,
    # then permit WARNING only within the remaining budget. Include >1.
    y=frame.risk_label.to_numpy(); normal=y=='NORMAL'; critical=y=='CRITICAL'
    critical_grid=np.unique(np.r_[0, np.quantile(prob[:,2],np.linspace(0,1,151)), 1.0000001])
    best=None
    for tc in critical_grid:
        flag=prob[:,2]>=tc; fpr=float(flag[normal].mean())
        if fpr>cap+1e-12: continue
        rec=float(flag[critical].mean())
        prec=float(critical[flag].mean()) if flag.any() else 0.
        key=(rec,prec,-fpr)
        if best is None or key>best[0]: best=(key,float(tc))
    tc=best[1]; cf=prob[:,2]>=tc
    best_w=None
    for tw in np.unique(np.r_[0,np.quantile(prob[:,1],np.linspace(0,1,101)),1.0000001]):
        pred=threshold_predictions(prob,dict(critical=tc,warning=float(tw)))
        if np.mean(pred[normal]!='NORMAL')>cap+1e-12: continue
        score=float(f1_score(y,pred,labels=list(CLASSES),average='macro',zero_division=0))
        key=(score,float(tw))
        if best_w is None or key>best_w[0]:best_w=(key,float(tw))
    return dict(critical=tc,warning=best_w[1])


def feature_sets(all_features):
    displacement=['displacement','rolling_mean','rolling_std','rolling_min','rolling_max','slope','velocity','acceleration','trend','persistence','change_point_score']
    health=['packet_loss','missing_ratio','stuck_sensor_flag','drift_score']
    return {'original':all_features, 'displacement_health':displacement+health,
            'displacement_physics':displacement+health+['expected_displacement','physics_residual','physics_residual_velocity']}


def train_risk(parts, params, features):
    fit,cal,decision=parts
    p=dict(params); mode=p.pop('weighting','none'); factor=p.pop('critical_weight',1.)
    method=p.pop('calibration','none');p.pop('feature_set',None)
    y=fit.risk_label.map({c:i for i,c in enumerate(CLASSES)}).to_numpy()
    counts=np.bincount(y,minlength=3)
    weights=(len(y)/(3*counts))[y]**{'none':0,'sqrt':.5,'inverse':1}[mode]
    weights[y==2]*=factor
    model=XGBClassifier(**p,objective='multi:softprob',num_class=3,eval_metric='mlogloss',tree_method='hist',random_state=SEED,n_jobs=1)
    model.fit(fit[features].to_numpy(float),y,sample_weight=weights)
    calibration=None if method=='none' else _calibrate(model.predict_proba(cal[features].to_numpy(float)),cal.risk_label.to_numpy(),method)
    bundle=OptimizedRiskModel(model,features,calibration,{})
    bundle.thresholds=select_thresholds(decision,bundle.predict_proba(decision),FPR_CAP)
    return bundle


def train_if(parts, params, features):
    fit,cal,decision=parts
    p=dict(params);p.pop('feature_set',None)
    model=IsolationForest(**p,contamination='auto',random_state=SEED,n_jobs=1)
    model.fit(fit.loc[healthy_baseline_mask(fit),features].to_numpy(float))
    score=-model.score_samples(decision[features].to_numpy(float))
    negatives=decision.anomaly_label.to_numpy()==0
    # Use all anomaly-negative windows for an honest anomaly FPR constraint.
    threshold=float(np.nextafter(np.quantile(score[negatives],1-FPR_CAP,method='higher'),np.inf))
    return {'model':model,'features':features,'threshold':threshold}


def if_metrics(frame, score, threshold):
    y=frame.anomaly_label.to_numpy(int);pred=score>=threshold
    return dict(anomaly_recall=float(pred[y==1].mean()),anomaly_fpr=float(pred[y==0].mean()),
                anomaly_pr_auc=float(average_precision_score(y,score)), anomaly_macro_f1=float(f1_score(y,pred,average='macro')),rows=len(frame))


def mean_metrics(records):
    keys=[k for k,v in records[0].items() if isinstance(v,(int,float)) and k!='rows']
    return {k:float(np.mean([r[k] for r in records])) for k in keys}


def quality(m,kind):
    if kind=='risk':
        return m['critical_recall'] + .15*m['macro_pr_auc']+.1*m['macro_f1']-8*max(0,m['normal_fpr']-FPR_CAP)
    return m['anomaly_recall']+.15*m['anomaly_pr_auc']-8*max(0,m['anomaly_fpr']-FPR_CAP)


def evaluate_cv(plan,params,features,kind,trial=None):
    records=[]; chunks=[]
    for i,(parts,va) in enumerate(plan):
        if kind=='risk':
            model=train_risk(parts,params,features);prob=model.predict_proba(va);pred=model.predict(va)
            m=metrics(va,prob,pred)
            chunk=pd.DataFrame({'index':va.index,'p0':prob[:,0],'p1':prob[:,1],'p2':prob[:,2],'prediction':pred})
        else:
            model=train_if(parts,params,features);score=-model['model'].score_samples(va[features].to_numpy(float))
            m=if_metrics(va,score,model['threshold']);chunk=pd.DataFrame({'index':va.index,'score':score})
        m['fold']=i;records.append(m);chunks.append(chunk)
        if trial is not None:
            trial.report(quality(mean_metrics(records),kind),i)
            trial.set_user_attr('fold_metrics',records)
            if trial.should_prune():raise optuna.TrialPruned()
    return mean_metrics(records),records,pd.concat(chunks).sort_values('index')


def risk_suggest(t):
    return dict(n_estimators=t.suggest_int('n_estimators',150,650,step=100),max_depth=t.suggest_int('max_depth',2,6),
      learning_rate=t.suggest_float('learning_rate',.025,.18,log=True),min_child_weight=t.suggest_float('min_child_weight',2,40,log=True),
      gamma=t.suggest_float('gamma',0,4),subsample=t.suggest_float('subsample',.6,1),colsample_bytree=t.suggest_float('colsample_bytree',.65,1),
      colsample_bylevel=t.suggest_float('colsample_bylevel',.75,1),reg_alpha=t.suggest_float('reg_alpha',1e-5,5,log=True),
      reg_lambda=t.suggest_float('reg_lambda',.1,30,log=True),max_delta_step=t.suggest_float('max_delta_step',0,5),
      max_bin=t.suggest_categorical('max_bin',[128,256]),weighting=t.suggest_categorical('weighting',['none','sqrt','inverse']),
      critical_weight=t.suggest_float('critical_weight',1,4),calibration=t.suggest_categorical('calibration',CALIBRATION_CHOICES),
      feature_set=t.suggest_categorical('feature_set',FEATURE_CHOICES))


def if_suggest(t):
    return dict(n_estimators=t.suggest_int('n_estimators',100,300,step=50),max_samples=t.suggest_categorical('max_samples',[128,256,512,1024]),
        max_features=t.suggest_float('max_features',.5,1),bootstrap=t.suggest_categorical('bootstrap',[False,True]),
        feature_set=t.suggest_categorical('feature_set',FEATURE_CHOICES))


def optimize(plan,sets,kind,cfg):
    study=optuna.create_study(study_name=kind,storage=f'sqlite:///{OUT / "studies.sqlite3"}',load_if_exists=True,direction='maximize',
      sampler=optuna.samplers.TPESampler(seed=SEED),pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=1))
    def objective(t):
        p=risk_suggest(t) if kind=='risk' else if_suggest(t)
        m,records,_=evaluate_cv(plan,p,sets[p['feature_set']],kind,t)
        t.set_user_attr('metrics',m)
        return quality(m,kind)
    remaining=max(0,cfg['trials_per_model']-len(study.trials))
    if remaining:study.optimize(objective,n_trials=remaining,timeout=cfg['timeout_per_model_seconds'])
    dump(OUT/f'{kind}_study.json',[dict(number=t.number,state=t.state.name,value=t.value,params=t.params,attrs=t.user_attrs) for t in study.trials])
    return study.best_params


def alerts(frame,prob,threshold=.5,persistence=3):
    """Actual AlertEngine sweep; higher transitions retain spatial gates."""
    from copy import deepcopy
    from src.config import alerts_config
    from src.risk.alert_engine import AlertEngine
    cfg=deepcopy(alerts_config()['escalation'])
    cfg['GREEN_to_WATCH'].update(class_threshold=threshold,persistence_windows=persistence)
    eng=AlertEngine(escalation=cfg)
    states=np.empty(len(frame),object);pos=pd.Series(np.arange(len(frame)),index=frame.index)
    for _,g in frame.groupby('event_id',sort=False):
        for idx,row in g.sort_values('window_timestamp').iterrows():
            i=int(pos.loc[idx]);states[i]=eng.update(f'{row.event_id}/{row.node_id}',dict(zip(CLASSES,map(float,prob[i]))),conditions={'neighbour_confirmations':0,'spatial_coherence_above_threshold':False})
    normal_events=critical_events=normal_alerts=detected=0;delays=[];episodes=0;days=0.
    result=frame[['event_id','window_timestamp','risk_label']].copy();result['state']=states
    for _,g in result.groupby('event_id',sort=False):
        active=g.state.to_numpy()!='GREEN'
        if (g.risk_label=='NORMAL').all():
            normal_events+=1;normal_alerts+=int(active.any());episodes+=int(np.sum(active & ~np.r_[False,active[:-1]]))
            days+=(float(g.window_timestamp.max())-float(g.window_timestamp.min()))/24
        if (g.risk_label=='CRITICAL').any():
            critical_events+=1
            if active.any():detected+=1;delays.append(float(g.loc[g.state!='GREEN','window_timestamp'].min()))
    return dict(watch_threshold=threshold,persistence_windows=persistence,normal_event_alert_fraction=normal_alerts/max(normal_events,1),
       critical_event_watch_recall=detected/max(critical_events,1),critical_events=critical_events,detected_critical_events=detected,
       median_watch_delay_from_scenario_start_hours=float(np.median(delays)) if delays else None,
       lead_time_hours=None,lead_time_reason='risk is constant over an event; no collapse/onset timestamp is available',
       false_alert_episodes=episodes,false_alerts_per_observed_normal_event_day=episodes/days if days else None,
       state_counts=pd.Series(states).value_counts().to_dict(),higher_escalation='gated: no co-temporal neighbors',recovery='unsupported by current AlertEngine')


def diagnostics(frame,prob,pred,bundle):
    y=frame.risk_label.to_numpy();rows=[]
    for fam,g in frame.groupby('scenario_family'):
        ix=frame.index.get_indexer(g.index);mask=g.risk_label=='CRITICAL'
        rows.append(dict(scenario=fam,rows=len(g),normal_fpr=float(np.mean(pred[ix][g.risk_label=='NORMAL']!='NORMAL')) if (g.risk_label=='NORMAL').any() else None,
          critical_recall=float(np.mean(pred[ix][mask]=='CRITICAL')) if mask.any() else None))
    pd.DataFrame(rows).to_csv(OUT/'scenario_errors.csv',index=False)
    frame.groupby('scenario_family')[bundle.features].agg(['mean','std']).to_csv(OUT/'feature_group_distributions.csv')
    fig,ax=plt.subplots();cm=confusion_matrix(y,pred,labels=list(CLASSES));im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='Actual',title='Reserved development validation')
    fig.tight_layout();fig.savefig(OUT/'validation_confusion_matrix.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots()
    for i,c in enumerate(CLASSES):
        p,r,_=precision_recall_curve(y==c,prob[:,i]);ax.plot(r,p,label=c)
    ax.legend();ax.set(xlabel='Recall',ylabel='Precision',title='Development precision–recall');fig.tight_layout();fig.savefig(OUT/'precision_recall.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots();normal=y=='NORMAL';crit=y=='CRITICAL';points=[]
    for t in np.unique(np.r_[0,np.quantile(prob[:,2],np.linspace(0,1,201)),1.0000001]):
        flag=prob[:,2]>=t;points.append(dict(threshold=float(t),critical_recall=float(flag[crit].mean()),normal_critical_fpr=float(flag[normal].mean())))
    curve=pd.DataFrame(points);curve.to_csv(OUT/'critical_recall_curve.csv',index=False)
    ax.plot(curve.normal_critical_fpr,curve.critical_recall);ax.axvline(.05,color='red',linestyle='--');ax.set(xlabel='NORMAL → CRITICAL FPR (diagnostic sweep only)',ylabel='Critical recall',title='Reserved development validation');fig.tight_layout();fig.savefig(OUT/'critical_recall_curve.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots()
    for i,c in enumerate(CLASSES):
        obs,p=calibration_curve(y==c,prob[:,i],n_bins=10,strategy='quantile');ax.plot(p,obs,'o-',label=c)
    ax.plot([0,1],[0,1],'k--');ax.legend();ax.set(xlabel='Predicted probability',ylabel='Observed fraction',title='Reserved development calibration');fig.tight_layout();fig.savefig(OUT/'calibration_curve.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots()
    for c in CLASSES:ax.hist(prob[y==c,2],bins=30,alpha=.4,density=True,label=c)
    ax.legend();ax.set(xlabel='P(CRITICAL)',ylabel='Density');fig.tight_layout();fig.savefig(OUT/'critical_probability_distribution.png',dpi=140);plt.close(fig)


def main():
    global OUT, SEED, FPR_CAP, FEATURE_CHOICES, CALIBRATION_CHOICES
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/optimization.yaml')
    ap.add_argument('--output-dir',default='reports/optimization');args=ap.parse_args()
    OUT=(ROOT/args.output_dir).resolve()
    cfg=yaml.safe_load((ROOT/args.config).read_text());OUT.mkdir(parents=True,exist_ok=True)
    SEED=int(cfg['seed']);FPR_CAP=float(cfg['normal_fpr_cap'])
    FEATURE_CHOICES=cfg['feature_sets'];CALIBRATION_CHOICES=cfg['calibration_methods']
    if (OUT/'freeze.json').exists():raise SystemExit('candidate already frozen; use a separate checkout/output directory for a new run')
    frame,all_features=development_frame();sets=feature_sets(all_features)
    train=frame[frame.split=='train'].copy();validation=frame[frame.split=='validation'].copy()
    assert not set(train.generation_parameter_id)&set(validation.generation_parameter_id)
    dump(OUT/'audit.json',dict(development_rows=len(frame),train_rows=len(train),validation_rows=len(validation),
       classes_by_scenario=pd.crosstab(frame.scenario_family,frame.risk_label).to_dict(),train_parameter_groups=train.generation_parameter_id.nunique(),
       validation_parameter_groups=validation.generation_parameter_id.nunique(),feature_sets=sets,test_opened=False))
    plan=folds(train,cfg['folds']);pd.DataFrame([dict(event_id=e,outer_fold=i) for i,(_,v) in enumerate(plan) for e in v.event_id.unique()]).to_csv(OUT/'fold_assignment.csv',index=False)
    original=yaml.safe_load((ROOT/'configs/model_params.yaml').read_text())
    bp=dict(original['risk_classifier']['params']);bp['weighting']=bp.pop('class_weighting');bp.pop('critical_threshold');bp.pop('warning_threshold')
    bp.update(n_estimators=original['risk_classifier']['n_estimators'],calibration=original['risk_classifier']['calibration']['method'],feature_set='original')
    baseline_if=dict(original['isolation_forest']['params'],feature_set='original')
    records=[]
    def record(name,kind,p,fs):
        print(f'Evaluating {name}',flush=True);m,fold_metrics,oof=evaluate_cv(plan,p,fs,kind)
        records.append(dict(experiment=name,scope='train_grouped_cv_selection',**m));dump(OUT/f'{name}_folds.json',fold_metrics)
        oof.to_csv(OUT/f'{name}_oof.csv',index=False);pd.DataFrame(records).to_csv(OUT/'experiment_log.csv',index=False)
        return m
    record('original_config_refit','risk',bp,all_features)
    record('original_if_refit','if',baseline_if,all_features)
    best=optimize(plan,sets,'risk',cfg);ibest=optimize(plan,sets,'if',cfg)
    record('optimized_risk','risk',best,sets[best['feature_set']]);record('optimized_if','if',ibest,sets[ibest['feature_set']])
    # Fixed ablation candidates are diagnostics only, not another unbounded search.
    for name in sets:
        if name=='original' or name==best['feature_set']:continue
        p=dict(best,feature_set=name);record('ablation_'+name,'risk',p,sets[name])
    parts=inner_parts(train)
    risk=train_risk(parts,best,sets[best['feature_set']]);baseline=train_risk(parts,bp,all_features)
    anom=train_if(parts,ibest,sets[ibest['feature_set']]);ibase=train_if(parts,baseline_if,all_features)
    # Thresholds for alternate FPR caps fitted ONLY on decision partition.
    decision=parts[2];dp=risk.predict_proba(decision)
    operating={str(cap):select_thresholds(decision,dp,cap) for cap in [.01,.05,.10,.20]}
    alert_rows=[]
    for t in [.3,.5,.7,.9]:
        for persistence in [1,2,3]:alert_rows.append(alerts(decision,dp,t,persistence))
    eligible=[r for r in alert_rows if r['normal_event_alert_fraction']<=.05]
    alert_selected=max(eligible or alert_rows,key=lambda r:(r['critical_event_watch_recall']-8*max(0,r['normal_event_alert_fraction']-.05),-(r['median_watch_delay_from_scenario_start_hours'] or 0)))
    dump(OUT/'alert_development_sweep.json',dict(selection_partition='inner decision groups only',candidates=alert_rows,selected=alert_selected))
    bundles=dict(risk=risk,baseline=baseline,anomaly=anom,baseline_anomaly=ibase,alert=alert_selected)
    model_path=(ROOT/'models/optimization/candidate.joblib') if OUT==(ROOT/'reports/optimization') else OUT/'candidate.joblib';model_path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(bundles,model_path)
    saved=dict(config=cfg,risk=best,isolation_forest=ibest,risk_features=risk.features,anomaly_features=anom['features'],thresholds=risk.thresholds,
      anomaly_threshold=anom['threshold'],operating_points=operating,alert=alert_selected,fit_rows=len(parts[0]),calibration_rows=len(parts[1]),threshold_rows=len(parts[2]))
    (OUT/'best_development_config.yaml').write_text(yaml.safe_dump(saved,sort_keys=False))
    dump(OUT/'freeze.json',dict(model_path=str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path),model_sha256=sha(model_path),config_sha256=sha(OUT/'best_development_config.yaml'),source_config_sha256=sha(ROOT/args.config),
         feature_store_sha256=sha(ROOT/'data/features/features_v2.parquet'),split_sha256=sha(ROOT/'data/features/split_assignment.csv'),test_generated=False,
         seed=SEED,selection='train parameter-grouped CV; validation excluded from all selection'))
    # From here onward: no model, threshold or alert selection.
    comparison=[]
    for name,model in [('original_config_refit',baseline),('optimized_risk',risk)]:
        prob=model.predict_proba(validation);pred=model.predict(validation);comparison.append(dict(experiment=name,scope='reserved_development_validation',**metrics(validation,prob,pred)))
    for name,model in [('original_if_refit',ibase),('optimized_if',anom)]:
        score=-model['model'].score_samples(validation[model['features']].to_numpy(float));comparison.append(dict(experiment=name,scope='reserved_development_validation',**if_metrics(validation,score,model['threshold'])))
    prob=risk.predict_proba(validation);pred=risk.predict(validation)
    diagnostics(validation,prob,pred,risk)
    tradeoffs=[]
    for cap,thresholds in operating.items():tradeoffs.append(dict(fit_fpr_cap=cap,thresholds=thresholds,**metrics(validation,prob,threshold_predictions(prob,thresholds))))
    dump(OUT/'operating_points.json',tradeoffs)
    dump(OUT/'alert_validation.json',dict(original=alerts(validation,baseline.predict_proba(validation)),optimized=alerts(validation,prob,alert_selected['watch_threshold'],alert_selected['persistence_windows'])))
    # Label-independent perturbations of reserved development features. No retuning.
    stress=[];rng=np.random.default_rng(SEED)
    scale=parts[0][risk.features].std().replace(0,1)
    for level in [.01,.05,.1]:
        changed=validation.copy();changed[risk.features]+=rng.normal(size=(len(changed),len(risk.features)))*scale.to_numpy()*level
        pr=risk.predict_proba(changed);stress.append(dict(perturbation_std_fraction=level,**metrics(changed,pr,risk.predict(changed))))
    dump(OUT/'robustness.json',dict(note='feature-noise sensitivity, not physically coupled raw-sensor simulation',results=stress))
    # Permute whole nine-window event blocks; preserve within-event ordering.
    importance=[];base_ap=metrics(validation,prob,pred)['macro_pr_auc'];groups=list(validation.groupby('event_id',sort=True).indices.values())
    if len(set(map(len,groups)))==1:
        for feature in risk.features:
            changed=validation.copy();values=changed[feature].to_numpy(copy=True);order=rng.permutation(len(groups))
            for i,j in enumerate(order):values[groups[i]]=validation[feature].to_numpy()[groups[j]]
            changed[feature]=values;pr=risk.predict_proba(changed)
            importance.append(dict(feature=feature,macro_pr_auc_drop=base_ap-metrics(validation,pr,risk.predict(changed))['macro_pr_auc']))
    pd.DataFrame(importance).to_csv(OUT/'permutation_importance.csv',index=False)
    timings=[]
    for name,model in [('original_config_refit',baseline),('optimized_risk',risk)]:
        sample=validation.iloc[:1];model.predict(sample);elapsed=[]
        for _ in range(100):
            start=time.perf_counter();model.predict(sample);elapsed.append((time.perf_counter()-start)*1000)
        timings.append(dict(model=name,median_ms=float(np.median(elapsed)),p95_ms=float(np.quantile(elapsed,.95)),scope='one feature row including calibration and pandas; excludes extraction and transport'))
    dump(OUT/'latency.json',timings)
    pd.DataFrame(records+comparison).to_csv(OUT/'model_comparison.csv',index=False);dump(OUT/'validation_metrics.json',comparison)
    trial_rows=[]
    for kind in ['risk','if']:
        for trial in json.loads((OUT/f'{kind}_study.json').read_text()):
            trial_rows.append(dict(experiment=f'{kind}_trial_{trial["number"]}',scope='train_grouped_cv_selection',state=trial['state'],objective=trial['value'],parameters=json.dumps(trial['params'],sort_keys=True),**trial['attrs'].get('metrics',{})))
    pd.DataFrame(records+trial_rows).to_csv(OUT/'experiment_log.csv',index=False)
    print(json.dumps(comparison,indent=2),flush=True)

if __name__=='__main__':main()
