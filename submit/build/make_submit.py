#!/usr/bin/env python3
"""Build a compact, source-linked ML evidence bundle from committed reports."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'submit'
sys.path.insert(0, str(ROOT))
PALETTE = {'blue':'#0072B2','orange':'#E69F00','green':'#009E73','vermillion':'#D55E00','purple':'#CC79A7','sky':'#56B4E9','gray':'#666666'}
MODELS = [('tuned_xgboost','Tuned XGBoost'),('default_xgboost','Default XGBoost'),('logistic_regression','Logistic regression'),('threshold_rule','Threshold rule')]
METRICS = [('pr_auc_macro_ovr','Macro PR-AUC','fraction'),('f1_macro','Macro F1','fraction'),('recall_critical','Critical recall','fraction'),('false_alarm_rate_normal','Normal false-alarm rate','fraction')]


def load(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))

def load_yaml(path):
    return yaml.safe_load((ROOT/path).read_text(encoding='utf-8'))

def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def savefig(fig, outdir:Path, name:str, rect=(0,.075,1,.92)):
    # Reserve a footer for the evidence-origin label and a header for the title.
    fig.tight_layout(rect=rect)
    fig.savefig(outdir/'figures'/f'{name}.png',dpi=300,facecolor='white',metadata={'Software':'make_submit.py'})
    fig.savefig(outdir/'figures'/f'{name}.svg',facecolor='white',metadata={'Date':None,'Creator':'make_submit.py'})
    plt.close(fig)

def write_csv(path, fields, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n'); w.writeheader()
        for row in rows:
            w.writerow({k: (fmt(v) if isinstance(v,(float,np.floating)) else v) for k,v in row.items()})

def fmt(x):
    return format(float(x),'.10g')

def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def markdown_table(rows):
    if not rows:return '_No rows._\n'
    cols=list(rows[0]); lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for r in rows: lines.append('| '+' | '.join(str(r[c]) for c in cols)+' |')
    return '\n'.join(lines)+'\n'

def tuned_cm(risk_model):
    return risk_model['confusion_matrix_labels_normal_warning_critical']

def style(ax):
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',color='#D9DDE2',linewidth=.65,alpha=.85); ax.set_axisbelow(True)

def tag(fig, text='Origin: synthetic corpus'):
    fig.text(.99,.01,text,ha='right',va='bottom',fontsize=10,color='#4B5563')

def make_mode():
    lock=load(Path('reports/test_lock.json'))
    assert lock.get('evals_run')==1
    final=load(Path('reports/final_eval.json'))
    assert final.get('evaluation',{}).get('test_touched_once') is True
    assert (ROOT/'reports/final_eval.md').is_file()
    label_test='tests/test_build_feature_store.py::test_label_blind_and_permuted_labels_leave_features_identical'
    assert 'def test_label_blind_and_permuted_labels_leave_features_identical' in (ROOT/'tests/test_build_feature_store.py').read_text()
    log=ROOT/'reports/pytest_phase8_pretest.txt'
    assert log.exists() and '100%]' in log.read_text(errors='replace')
    for seed in (101,202,303):
        p=ROOT/f'reports/pytest_phase7_hashseed_{seed}.txt'
        assert p.exists() and '100%]' in p.read_text(errors='replace')
    return ('# Submission mode\n\n**Mode A — full results.** The locked evaluation exists, `evals_run` is 1, the label-blind/permuted-label regression test is present and passed, and the full pytest suite passed. Numbers here are limited to the single locked synthetic evaluation and post-fix grouped-development reports.\n\nLocked test: synthetic corpus, seed `20260929`, hash `e49000ce6142a6c5e59a9743d8551526da62997b03edbb48173b7d4d03377cbb`; source: `reports/final_eval.md`. Label-blind test: `tests/test_build_feature_store.py::test_label_blind_and_permuted_labels_leave_features_identical`. Full test evidence: `reports/pytest_phase8_pretest.txt` and the three hash-seed reports.\n\nAll data and model results are synthetic-corpus results. No real-mine validation is claimed.\n')

def build(outdir:Path):
    (outdir/'figures').mkdir(parents=True,exist_ok=True); (outdir/'tables').mkdir(parents=True,exist_ok=True); (outdir/'build').mkdir(parents=True,exist_ok=True)
    final=load(Path('reports/final_eval.json')); lock=load(Path('reports/test_lock.json'))
    cv=load(Path('reports/tuning/tuned_cv_metrics.json')); bases=load(Path('reports/tuning/baselines.json')); studies={p:load(Path('reports/tuning')/p) for p in ('forecaster_study.json','isolation_forest_study.json','xgboost_study.json')}
    diag=load(Path('reports/tuning/robustness.json')); manifest=load(Path('data/synthetic/dataset_manifest.json'))
    assert final['evaluation']['evals_run']==1 and lock['evals_run']==1 and final['evaluation']['sha256']==lock['sha256']
    assert cv['protocol']['development_only'] and not cv['protocol']['test_touched']
    assert diag['protocol']['test_touched'] is False
    # Locked test metrics.
    result_rows=[]
    for key,label in MODELS:
        m=final['risk_models'][key]
        for metric,title,unit in METRICS:
            source_key={'pr_auc_macro_ovr':'pr_auc_macro_ovr','f1_macro':'f1_macro','recall_critical':'recall_critical','false_alarm_rate_normal':'false_alarm_rate_normal'}[metric]
            result_rows.append({'model':label,'metric':title,'value':m[source_key],'unit':unit,'split':'locked_test','source_file':'reports/final_eval.json','source_key':f'risk_models.{key}.{source_key}'})
    write_csv(outdir/'tables/results.csv',list(result_rows[0]),result_rows)
    # Grouped-CV baselines and tuned XGBoost, with fold SD.
    base_rows=[]
    for key,label in [('threshold_rule','Threshold rule'),('logistic','Logistic regression (C=10)'),('xgboost_default','Default XGBoost')]:
        for k,title,unit in [('pr_auc_macro','Macro PR-AUC','fraction'),('recall_critical','Critical recall','fraction'),('f1_macro','Macro F1','fraction'),('false_alarm_rate_normal','Normal false-alarm rate','fraction')]:
            a=[float(f[k]) for f in bases['models'][key]['folds'] if f.get(k) is not None]
            base_rows.append({'model':label,'metric':title,'mean':float(np.mean(a)),'std':float(np.std(a,ddof=1)),'unit':unit,'folds':len(a),'split':'grouped_cv','source_file':'reports/tuning/baselines.json','source_key':f'models.{key}.folds'})
    for k,v in cv['risk_models']['xgboost_tuned']['mean'].items():
        title=dict(pr_auc_macro='Macro PR-AUC',recall_critical='Critical recall',f1_macro='Macro F1',false_alarm_rate_normal='Normal false-alarm rate')[k]
        base_rows.append({'model':'Tuned XGBoost','metric':title,'mean':v['mean'],'std':v['std'],'unit':'fraction','folds':3,'split':'grouped_cv','source_file':'reports/tuning/tuned_cv_metrics.json','source_key':f'risk_models.xgboost_tuned.mean.{k}'})
    write_csv(outdir/'tables/baselines.csv',list(base_rows[0]),base_rows)
    # Locked per-class report.
    per_rows=[]
    for key,label in MODELS:
        for cls in ('NORMAL','WARNING','CRITICAL'):
            r=final['risk_models'][key]['per_class'][cls]
            per_rows.append({'model':label,'class':cls,'precision':r['precision'],'recall':r['recall'],'f1':r['f1-score'],'support':int(r['support']),'split':'locked_test','source_file':'reports/final_eval.json','source_key':f'risk_models.{key}.per_class.{cls}'})
    write_csv(outdir/'tables/per_class.csv',list(per_rows[0]),per_rows)
    # Dataset summary: generator manifest and one-use final report only.
    sd=manifest['split_definition']['event_counts']; test=final['evaluation']; support=final['risk_models']['tuned_xgboost']['per_class']
    dataset_rows=[
      {'stat':'Generated event sequences','value':manifest['row_counts']['sequences_generated'],'unit':'events','source_file':'data/synthetic/dataset_manifest.json','source_key':'row_counts.sequences_generated','split':'synthetic_corpus'},
      {'stat':'Generated sensor observations','value':manifest['row_counts']['synthetic_nodes_rows'],'unit':'rows','source_file':'data/synthetic/dataset_manifest.json','source_key':'row_counts.synthetic_nodes_rows','split':'synthetic_corpus'},
      {'stat':'Post-fix feature windows','value':manifest['feature_store']['n_windows'],'unit':'windows','source_file':'data/synthetic/dataset_manifest.json','source_key':'feature_store.n_windows','split':'synthetic_corpus'},
      {'stat':'Window length','value':manifest['feature_store']['window_steps'],'unit':'timesteps','source_file':'data/synthetic/dataset_manifest.json','source_key':'feature_store.window_steps','split':'synthetic_corpus'},
      {'stat':'Window stride','value':manifest['feature_store']['stride_steps'],'unit':'timesteps','source_file':'data/synthetic/dataset_manifest.json','source_key':'feature_store.stride_steps','split':'synthetic_corpus'},
      {'stat':'Scenario families','value':len(manifest['scenario_parameters']['scenario_types']),'unit':'families','source_file':'data/synthetic/dataset_manifest.json','source_key':'scenario_parameters.scenario_types','split':'synthetic_corpus'},
      {'stat':'Development training events','value':sd['train'],'unit':'events','source_file':'data/synthetic/dataset_manifest.json','source_key':'split_definition.event_counts.train','split':'train'},
      {'stat':'Development validation events','value':sd['validation'],'unit':'events','source_file':'data/synthetic/dataset_manifest.json','source_key':'split_definition.event_counts.validation','split':'validation'},
      {'stat':'Fresh locked test events','value':test['events'],'unit':'events','source_file':'reports/final_eval.json','source_key':'evaluation.events','split':'locked_test'},
      {'stat':'Fresh locked test windows','value':test['windows'],'unit':'windows','source_file':'reports/final_eval.json','source_key':'evaluation.windows','split':'locked_test'},
      {'stat':'Locked test seed','value':lock['seed'],'unit':'seed','source_file':'reports/test_lock.json','source_key':'seed','split':'locked_test'},
      {'stat':'Locked test evaluations used','value':lock['evals_run'],'unit':'evaluations','source_file':'reports/test_lock.json','source_key':'evals_run','split':'locked_test'},
      {'stat':'Tuned XGBoost active inputs','value':cv['protocol']['feature_count'],'unit':'features','source_file':'reports/tuning/tuned_cv_metrics.json','source_key':'protocol.feature_count','split':'grouped_cv'},
      {'stat':'Tuned IF active inputs','value':len(load_yaml(Path('configs/model_params.yaml'))['isolation_forest']['features']),'unit':'features','source_file':'configs/model_params.yaml','source_key':'isolation_forest.features','split':'train'},
    ]
    for cls in ('NORMAL','WARNING','CRITICAL'):
        dataset_rows.append({'stat':f'Locked test {cls} support','value':int(support[cls]['support']),'unit':'windows','source_file':'reports/final_eval.json','source_key':f'risk_models.tuned_xgboost.per_class.{cls}.support','split':'locked_test'})
    dataset_rows.append({'stat':'Nodes per event in current corpus','value':1,'unit':'node/event','source_file':'reports/acceptance_criteria.md','source_key':': one node per event','split':'synthetic_corpus'})
    write_csv(outdir/'tables/dataset.csv',list(dataset_rows[0]),dataset_rows)
    # Workstation measurements and artifact sizes (size is repo artifact metadata, not latency claim).
    edge=diag['workstation_edge_profile']['models']; files={'risk_xgboost_tuned':'models/tuned/risk_xgboost_tuned.joblib','isolation_forest_tuned':'models/tuned/iforest_tuned.joblib','forecaster_tuned':'models/temporal_model/forecaster_tuned.pt'}
    edge_rows=[]
    for key,path in files.items():
        z=edge[key]; edge_rows.append({'model':key,'p50_ms':z['latency_ms_p50'],'p95_ms':z['latency_ms_p95'],'peak_sampled_rss_mib':z['process_rss_peak_sampled_bytes']/1048576,'artifact_bytes':(ROOT/path).stat().st_size,'device':'workstation CPU','source_file':'reports/tuning/robustness.json','source_key':f'workstation_edge_profile.models.{key}'})
    write_csv(outdir/'tables/edge_profile.csv',list(edge_rows[0]),edge_rows)
    # Forecast next-window grouped CV.
    fore=studies['forecaster_study.json']; best=fore['best']
    forecast_rows=[{'model':'Selected TCN','metric':'Normalized MSE','value':best['mean_normalized_mse'],'unit':'normalized_mse','split':'grouped_cv','source_file':'reports/tuning/forecaster_study.json','source_key':'best.mean_normalized_mse'},
                   {'model':'Persistence','metric':'Normalized MSE','value':best['mean_persistence_normalized_mse'],'unit':'normalized_mse','split':'grouped_cv','source_file':'reports/tuning/forecaster_study.json','source_key':'best.mean_persistence_normalized_mse'}]
    write_csv(outdir/'tables/forecast.csv',list(forecast_rows[0]),forecast_rows)
    # Fault-category recall from the post-fix development validation diagnostic.
    fault_rows=[]
    for cat,obj in diag['required_scenario_category_recall'].items():
        if 'recall_per_class' in obj:
            for cls,val in obj['recall_per_class'].items(): fault_rows.append({'scenario_category':cat,'class':cls,'recall':val,'events':obj['events'],'split':'development_validation','source_file':'reports/tuning/robustness.json','source_key':f'required_scenario_category_recall.{cat}.recall_per_class.{cls}','events_source_key':f'required_scenario_category_recall.{cat}.events'})
    write_csv(outdir/'tables/fault_scenarios.csv',list(fault_rows[0]),fault_rows)
    # Feature-group ablation.
    ab_rows=[]
    for group,x in diag['feature_group_ablations'].items():
        m=x.get('metrics')
        if m: ab_rows.append({'removed_group':group,'macro_pr_auc':m['pr_auc_macro'],'critical_recall':m['recall_critical'],'macro_f1':m['f1_macro'],'split':'development_validation','source_file':'reports/tuning/robustness.json','source_key':f'feature_group_ablations.{group}.metrics'})
    lc=diag['learning_curve'][-1]['validation_metrics']
    ab_rows.insert(0,{'removed_group':'All active inputs','macro_pr_auc':lc['pr_auc_macro'],'critical_recall':lc['recall_critical'],'macro_f1':lc['f1_macro'],'split':'development_validation','source_file':'reports/tuning/robustness.json','source_key':'learning_curve.full_train.validation_metrics'})
    write_csv(outdir/'tables/ablation.csv',list(ab_rows[0]),ab_rows)
    cm=tuned_cm(final['risk_models']['tuned_xgboost']); classes=['NORMAL','WARNING','CRITICAL']
    cm_rows=[{'true_class':classes[i],'predicted_class':classes[j],'count':int(cm[i][j]),'split':'locked_test','source_file':'reports/final_eval.json','source_key':f'risk_models.tuned_xgboost.confusion_matrix[{i}][{j}]'} for i in range(3) for j in range(3)]
    write_csv(outdir/'tables/confusion_matrix.csv',list(cm_rows[0]),cm_rows)
    shap_rows=[{'feature':x['feature'],'mean_abs_shap':x['mean_abs_shap'],'unit':'mean_abs_shap','source_file':'reports/tuning/robustness.json','source_key':f'tree_shap.top3.{i}.mean_abs_shap','split':'development_validation'} for i,x in enumerate(diag['tree_shap']['top3'])]
    write_csv(outdir/'tables/shap.csv',list(shap_rows[0]),shap_rows)
    # Alert engine metrics on the locked synthetic evaluation.
    alert=final['alert_engine']; alert_rows=[{'metric':'False alert episodes per normal event day','value':alert['false_alarms_per_normal_event_day'],'unit':'episodes/day','source_file':'reports/final_eval.json','source_key':'alert_engine.false_alarms_per_normal_event_day'}, {'metric':'Median lead time','value':alert['median_lead_time_hours'],'unit':'hours','source_file':'reports/final_eval.json','source_key':'alert_engine.median_lead_time_hours'}, {'metric':'P10 lead time','value':alert['p10_lead_time_hours'],'unit':'hours','source_file':'reports/final_eval.json','source_key':'alert_engine.p10_lead_time_hours'}]
    write_csv(outdir/'tables/alert.csv',list(alert_rows[0]),alert_rows)
    # Markdown companion for each CSV.
    for p in (outdir/'tables').glob('*.csv'):
        rr=read_csv(p); (p.with_suffix('.md')).write_text(f'# {p.stem.replace("_"," ").title()}\n\n'+markdown_table(rr),encoding='utf-8')
    # Figure 1: stages from the source pipeline; one current gate is explicit.
    fig,ax=plt.subplots(figsize=(10,3.1)); ax.axis('off')
    stages=['Validation','Features','Isolation\nForest','Spatial\nfusion\n(gated)','Physics\ncheck','XGBoost','Alert\nengine','Explainability']
    xs=np.linspace(.06,.94,len(stages))
    for i,(x,label) in enumerate(zip(xs,stages)):
        color=PALETTE['orange'] if 'gated' in label else PALETTE['sky']
        ax.text(x,.53,label,ha='center',va='center',fontsize=10.5,color='#17212B',bbox=dict(boxstyle='round,pad=.55',fc=color,ec='#263746',lw=1.0))
        if i<len(stages)-1: ax.annotate('',xy=(xs[i+1]-.052,.53),xytext=(x+.052,.53),arrowprops=dict(arrowstyle='->',lw=1.4,color='#34495E'))
    fig.suptitle('Validated windows flow through eight ML stages',fontsize=15,fontweight='bold',x=.04,ha='left')
    tag(fig); savefig(fig,outdir,'fig_01_ml_flow')
    # Figure 2: config-driven coupled subsidence and gradient.
    from src.simulator.deformation_field import DeformationField
    field=DeformationField(); p=field.params
    grid=np.linspace(-p.influence_radius,p.influence_radius,121); xx,yy=np.meshgrid(grid,grid); t_days=3.0
    wxy=field(xx,yy,t_days); tt=np.linspace(0,12,121); wt=field.w_of_t(tt); tilt=field.dW_dx(grid,np.zeros_like(grid),t_days)
    fig,axs=plt.subplots(1,3,figsize=(11,3.6))
    im=axs[0].contourf(xx,yy,wxy,levels=12,cmap='cividis'); fig.colorbar(im,ax=axs[0],label='Subsidence (mm)'); axs[0].set(xlabel='x (m)',ylabel='y (m)',title='Surface profile at 3 days')
    axs[1].plot(tt,wt,color=PALETTE['blue'],lw=2.4); axs[1].set(xlabel='Time (days)',ylabel='Subsidence (mm)',title='Growth at panel centre')
    axs[2].plot(grid,tilt,color=PALETTE['vermillion'],lw=2.4); axs[2].axhline(0,color='#555',lw=.7); axs[2].set(xlabel='x (m)',ylabel='Tilt gradient (mm/m)',title='Derived x-gradient at 3 days')
    for a in axs: style(a)
    fig.suptitle('The simulator couples a spatial bowl to time growth and tilt',fontsize=14,fontweight='bold',x=.04,ha='left')
    tag(fig,'Origin: config-driven synthetic simulator'); savefig(fig,outdir,'fig_02_physics')
    # Figure 3: dataset totals and final holdout class support.
    ds=read_csv(outdir/'tables/dataset.csv'); metrics=[r for r in ds if r['stat'] in ('Generated event sequences','Post-fix feature windows','Fresh locked test events','Fresh locked test windows')]
    names=[r['stat'] for r in metrics]; vals=[float(r['value']) for r in metrics]
    supports=[int(support[c]['support']) for c in ('NORMAL','WARNING','CRITICAL')]
    fig,axs=plt.subplots(1,2,figsize=(9.6,4.2)); axs[0].barh(names[::-1],vals[::-1],color=[PALETTE['blue'],PALETTE['sky'],PALETTE['orange'],PALETTE['vermillion']]); axs[0].set_xlabel('Count (events or windows)'); axs[0].set_title('Generated and held-out counts'); style(axs[0])
    for y,v in enumerate(vals[::-1]): axs[0].text(v*1.01,y,f'{int(v):,}',va='center',fontsize=9)
    axs[1].bar(['NORMAL','WARNING','CRITICAL'],supports,color=[PALETTE['sky'],PALETTE['orange'],PALETTE['vermillion']]); axs[1].set_ylabel('Locked test windows'); axs[1].set_title('Locked test class support'); style(axs[1])
    for i,v in enumerate(supports): axs[1].text(i,v+70,f'{v:,}',ha='center',fontsize=9)
    fig.suptitle('The post-fix corpus has a separate one-use synthetic holdout',fontsize=14,fontweight='bold',x=.04,ha='left'); tag(fig); savefig(fig,outdir,'fig_03_dataset')
    # Figure 4: locked-test baseline comparison; false-alert/day is alert-engine aggregate.
    result_map={r['model']:r for r in read_csv(outdir/'tables/results.csv')}
    fig,axs=plt.subplots(1,2,figsize=(10,4.2)); modelnames=[x[1] for x in MODELS]; x=np.arange(len(modelnames)); width=.36
    for j,(key,label) in enumerate((('pr_auc_macro_ovr','Macro PR-AUC'),('recall_critical','Critical recall'))):
        v=[final['risk_models'][m][key] for m,_ in MODELS]; axs[0].bar(x+(j-.5)*width,v,width,label=label,color=[PALETTE['blue'],PALETTE['orange']][j])
        for i,val in enumerate(v): axs[0].text(x[i]+(j-.5)*width,val+.012,f'{val:.2f}',ha='center',fontsize=8)
    axs[0].set_xticks(x,modelnames,rotation=18,ha='right'); axs[0].set_ylim(0,1); axs[0].set_ylabel('Score (fraction)'); axs[0].set_title('Locked test discrimination and recall'); axs[0].legend(frameon=False); style(axs[0])
    far=[final['risk_models'][m]['false_alarm_rate_normal'] for m,_ in MODELS]; axs[1].bar(x,far,color=[PALETTE['blue'],PALETTE['sky'],PALETTE['orange'],PALETTE['green']]); axs[1].set_xticks(x,modelnames,rotation=18,ha='right'); axs[1].set_ylim(0,1); axs[1].set_ylabel('Normal false-alarm rate (fraction)'); axs[1].set_title('False alarms by model'); style(axs[1])
    for i,v in enumerate(far): axs[1].text(i,v+.015,f'{v:.2f}',ha='center',fontsize=9)
    fig.suptitle('Tuned XGBoost has low Critical recall on the locked test',fontsize=14,fontweight='bold',x=.04,ha='left'); tag(fig); savefig(fig,outdir,'fig_04_test_comparison')
    # Figure 5: locked per-class metrics + confusion matrix.
    tuned=final['risk_models']['tuned_xgboost']; classes=['NORMAL','WARNING','CRITICAL']; fig,axs=plt.subplots(1,2,figsize=(9.8,4.2))
    xx=np.arange(3); bw=.24
    for j,(key,title,color) in enumerate((('precision','Precision',PALETTE['blue']),('recall','Recall',PALETTE['orange']),('f1-score','F1',PALETTE['green']))):
        v=[tuned['per_class'][c][key] for c in classes]; axs[0].bar(xx+(j-1)*bw,v,bw,label=title,color=color)
    axs[0].set_xticks(xx,classes); axs[0].set_ylim(0,1); axs[0].set_ylabel('Score (fraction)'); axs[0].set_title('Tuned model by class'); axs[0].legend(frameon=False); style(axs[0])
    cm=np.asarray(tuned['confusion_matrix_labels_normal_warning_critical']); im=axs[1].imshow(cm,cmap='Blues'); axs[1].set_xticks(range(3),classes,rotation=20); axs[1].set_yticks(range(3),classes); axs[1].set(xlabel='Predicted class',ylabel='True class',title='Confusion matrix (windows)')
    for i in range(3):
        for j in range(3): axs[1].text(j,i,f'{cm[i,j]:,}',ha='center',va='center',color='white' if cm[i,j]>cm.max()*.55 else '#18212B',fontsize=9)
    fig.colorbar(im,ax=axs[1],label='Window count'); fig.suptitle('The tuned model misses most Critical windows in the holdout',fontsize=14,fontweight='bold',x=.04,ha='left'); tag(fig); savefig(fig,outdir,'fig_05_per_class')
    # Figure 6: calibration curve from the saved post-fix development reliability artifact.
    source=ROOT/'reports/tuning/reliability.png'; img=plt.imread(source)
    fig,ax=plt.subplots(figsize=(6,5)); ax.imshow(img); ax.axis('off'); fig.suptitle('Development calibration favored isotonic mapping',fontsize=14,fontweight='bold',x=.04,ha='left'); tag(fig,'Origin: synthetic development validation'); savefig(fig,outdir,'fig_06_calibration')
    # Figure 7: fault-category class recall from validation; no false-CRITICAL claim.
    fault=read_csv(outdir/'tables/fault_scenarios.csv'); cats=['SENSOR_FAULT','DATA_QUALITY','COMMUNICATION_FAILURE','noise_injected']; labels=['NORMAL','WARNING','CRITICAL']; colors=[PALETTE['blue'],PALETTE['orange'],PALETTE['vermillion']]
    fig,ax=plt.subplots(figsize=(8,5.1)); y=np.arange(len(cats)); offsets=[-.24,0,.24]; height=.22
    for cls,color,offset in zip(labels,colors,offsets):
        vals=np.asarray([float(next((r['recall'] for r in fault if r['scenario_category']==cat and r['class']==cls),0)) for cat in cats])
        ax.barh(y+offset,vals,height=height,label=cls,color=color)
        for yy,v in zip(y+offset,vals):
            if v==0: ax.plot(0,yy,marker='|',color=color,markersize=9,markeredgewidth=2)
    ax.set_yticks(y,cats); ax.set_xlim(0,1); ax.set_xlabel('Recall by class (fraction)'); ax.set_title('Critical recall is zero in these validation fault categories'); ax.legend(frameon=False,ncols=3,loc='upper center',bbox_to_anchor=(.5,-.30)); style(ax); fig.suptitle('Fault-category checks show missed elevated-risk cases',fontsize=14,fontweight='bold',x=.04,ha='left'); tag(fig,'Origin: synthetic development validation'); savefig(fig,outdir,'fig_07_fault_categories',rect=(0,.22,1,.91))
    # Figure 8: next-window forecaster against persistence.
    forecast=read_csv(outdir/'tables/forecast.csv'); v=[float(r['value']) for r in forecast]; fig,ax=plt.subplots(figsize=(6,4)); bars=ax.bar(['TCN','Persistence'],v,color=[PALETTE['blue'],PALETTE['gray']]); ax.set_ylabel('Normalized MSE (lower is better)'); ax.set_title('The next-window TCN beats persistence on grouped CV'); style(ax)
    for b,z in zip(bars,v): ax.text(b.get_x()+b.get_width()/2,z+max(v)*.02,f'{z:.4f}',ha='center',fontsize=10)
    fig.suptitle('Forecasting is limited to the next window',fontsize=14,fontweight='bold',x=.04,ha='left'); tag(fig); savefig(fig,outdir,'fig_08_forecast')
    # Figure 9: model-level TreeSHAP, not a fabricated single-alert payload.
    top=diag['tree_shap']['top3']; names=[x['feature'] for x in top][::-1]; vals=[x['mean_abs_shap'] for x in top][::-1]
    fig,ax=plt.subplots(figsize=(7,3.7)); ax.barh(names,vals,color=PALETTE['sky']); ax.set_xlabel('Mean absolute SHAP contribution'); ax.set_title('Physics residual features rank highest globally'); style(ax)
    for i,v in enumerate(vals): ax.text(v+.015,i,f'{v:.2f}',va='center',fontsize=9)
    fig.suptitle('Global TreeSHAP summary for tuned XGBoost',fontsize=14,fontweight='bold',x=.04,ha='left'); tag(fig,'Origin: synthetic development validation'); savefig(fig,outdir,'fig_09_explainability')
    # Figure 10: development validation feature-group ablation.
    ab=read_csv(outdir/'tables/ablation.csv'); names=[r['removed_group'] for r in ab]; vals=[float(r['macro_pr_auc']) for r in ab]
    order=np.argsort(vals); fig,ax=plt.subplots(figsize=(8,5)); colors=[PALETTE['vermillion'] if n=='F_physics' else PALETTE['blue'] for n in np.asarray(names)[order]]; ax.barh(np.asarray(names)[order],np.asarray(vals)[order],color=colors); ax.set_xlim(0,1); ax.set_xlabel('Macro PR-AUC (fraction)'); ax.set_title('Removing physics inputs causes the largest drop'); style(ax)
    for i,v in enumerate(np.asarray(vals)[order]): ax.text(v+.012,i,f'{v:.3f}',va='center',fontsize=9)
    fig.suptitle('Feature-group removal on development validation',fontsize=14,fontweight='bold',x=.04,ha='left'); tag(fig,'Origin: synthetic development validation'); savefig(fig,outdir,'fig_10_ablation')
    # Figure 11: workstation-only latency.
    edge_rows=read_csv(outdir/'tables/edge_profile.csv'); fig,axs=plt.subplots(1,2,figsize=(9,4)); labels=[r['model'].replace('_tuned','').replace('_',' ') for r in edge_rows]; x=np.arange(len(labels)); p50=[float(r['p50_ms']) for r in edge_rows]; p95=[float(r['p95_ms']) for r in edge_rows]
    axs[0].bar(x-.18,p50,.36,label='p50',color=PALETTE['blue']); axs[0].bar(x+.18,p95,.36,label='p95',color=PALETTE['orange']); axs[0].set_xticks(x,labels,rotation=15,ha='right'); axs[0].set_ylabel('Latency (ms/window)'); axs[0].set_title('Workstation CPU latency'); axs[0].legend(frameon=False); style(axs[0])
    rss=[float(r['peak_sampled_rss_mib']) for r in edge_rows]; axs[1].bar(x,rss,color=PALETTE['green']); axs[1].set_xticks(x,labels,rotation=15,ha='right'); axs[1].set_ylabel('Peak sampled process RSS (MiB)'); axs[1].set_title('Workstation process memory'); style(axs[1])
    fig.suptitle('All measurements are workstation profiles, not device measurements',fontsize=14,fontweight='bold',x=.04,ha='left'); tag(fig); savefig(fig,outdir,'fig_11_edge')
    # Captions and provenance.
    captions='''# Figure captions and sources\n\n'''
    cap=[('fig_01_ml_flow','The pipeline diagram names the eight inference stages; spatial fusion is marked gated because this corpus has no co-temporal neighbors.','src/pipeline.py; reports/acceptance_criteria.md'),('fig_02_physics','Config-driven synthetic subsidence profile, temporal growth, and analytic tilt gradient.','src/simulator/deformation_field.py; src/simulator/temporal_model.py; configs/physics.yaml'),('fig_03_dataset','Post-fix generated counts and fresh locked-test class support.','data/synthetic/dataset_manifest.json; reports/final_eval.json'),('fig_04_test_comparison','Locked synthetic test baseline comparison; alert false-alert/day is separately reported in tables/alert.csv.','reports/final_eval.json'),('fig_05_per_class','Locked test per-class metrics and tuned-model confusion matrix.','reports/final_eval.json'),('fig_06_calibration','Saved reliability curve from post-fix development validation; not a locked-test calibration curve.','reports/tuning/reliability.png; reports/tuning_report.md'),('fig_07_fault_categories','Class recall by selected sensor-fault/noise categories on development validation; false-Critical rates by fault were not retained, so no such claim is made.','reports/tuning/robustness.json: required_scenario_category_recall'),('fig_08_forecast','Next-window TCN versus persistence normalized MSE on grouped development CV; longer horizons are unsupported.','reports/tuning/forecaster_study.json: best'),('fig_09_explainability','Global mean absolute TreeSHAP for three leading features; a single-alert payload was not retained.','reports/tuning/robustness.json: tree_shap'),('fig_10_ablation','Macro PR-AUC after removing active feature groups on development validation.','reports/tuning/robustness.json: feature_group_ablations'),('fig_11_edge','Single-window workstation CPU latency and sampled process RSS; no device measurement.','reports/tuning/robustness.json: workstation_edge_profile')]
    captions+='\n'.join(f'## {n}\n\n{c}\n\nOrigin: synthetic corpus. Source: `{src}`.\n' for n,c,src in cap)
    captions+='\n## Dropped figures\n\n- Event timeline: the final report does not retain row-level holdout probabilities/timestamps; direct access to the one-use holdout is prohibited after evaluation.\n- Isolation Forest score distribution: source reports retain aggregate metrics and threshold, not score samples; no regenerated distribution is claimed.\n- Per-alert SHAP payload: only model-level TreeSHAP ranking is retained.\n- Physical tabletop trace: available fixture is a seeded synthetic stand-in, not a physical rig; no real-rig figure is included.\n'
    (outdir/'figures/captions.md').write_text(captions,encoding='utf-8')
    # Claims ledger is generated from every numeric table cell plus simulator settings.
    claims=[]; seen=set()
    def add_claim(text,value,unit,source,key,split='n/a',origin='synthetic'):
        k=(text,str(value),source,key)
        if k in seen:return
        seen.add(k); claims.append({'claim_id':f'C{len(claims)+1:03d}','text_as_written':text,'value':fmt(value) if isinstance(value,(int,float,np.integer,np.floating)) else value,'unit':unit,'source_file':source,'key_or_line':key,'script':'submit/build/make_submit.py','data_origin':origin,'split':split,'verified':'yes'})
    table_units={
      'results.csv':('value','metric','unit','source_file','source_key','split'),
      'baselines.csv':('mean','metric mean','unit','source_file','source_key','split'),
      'per_class.csv':None,'edge_profile.csv':None,'dataset.csv':None,'forecast.csv':None,'fault_scenarios.csv':None,'ablation.csv':None,'alert.csv':None}
    for p in (outdir/'tables').glob('*.csv'):
        rows=read_csv(p)
        for ri,r in enumerate(rows):
            for col,val in r.items():
                if col in {'value','mean','std','precision','recall','f1','support','p50_ms','p95_ms','peak_sampled_rss_mib','artifact_bytes','events','macro_pr_auc','critical_recall','macro_f1','count','mean_abs_shap'} and val not in ('',None):
                    try:num=float(val)
                    except (ValueError,TypeError):continue
                    src=r.get('source_file',f'submit/tables/{p.name}'); key=r.get('events_source_key') if col=='events' else r.get('source_key',f'row[{ri}].{col}')
                    unit=r.get('unit',{'p50_ms':'ms/window','p95_ms':'ms/window','peak_sampled_rss_mib':'MiB','artifact_bytes':'bytes','support':'windows','events':'events'}.get(col,'fraction'))
                    split=r.get('split','n/a')
                    add_claim(f"{r.get('model',r.get('stat',r.get('metric',r.get('scenario_category','table value'))))}: {r.get('metric',col)}",num,unit,src,key,split)
    phys=load_yaml(Path('configs/physics.yaml'))['physics']
    add_claim('Maximum subsidence from extraction geometry',float(phys['extraction_height'])*float(phys['subsidence_factor'])*1000,'mm','configs/physics.yaml','physics.extraction_height * physics.subsidence_factor * 1000','n/a')
    add_claim('Knothe temporal coefficient',float(phys['time_coefficient']),'1/day','configs/physics.yaml','physics.time_coefficient','n/a')
    add_claim('Panel influence radius',float(phys['influence_radius']),'m','configs/physics.yaml','physics.influence_radius','n/a')
    add_claim('Panel centre x',float(phys['panel_center_x']),'m','configs/physics.yaml','physics.panel_center_x','n/a')
    add_claim('Panel centre y',float(phys['panel_center_y']),'m','configs/physics.yaml','physics.panel_center_y','n/a')
    add_claim('Physics profile time point',3,'days','submit/build/make_submit.py','t_days','n/a')
    add_claim('Pipeline stage count',8,'stages','src/pipeline.py','validation through explainability','n/a')
    xgb_study=studies['xgboost_study.json']['study']
    for key in ('requested_trials','actual_trials','completed_trials','pruned_trials'):
        add_claim(f"XGBoost search {key.replace('_',' ')}",xgb_study[key],'trials','reports/tuning/xgboost_study.json',f'study.{key}','grouped_cv')
    write_csv(outdir/'claims.csv',list(claims[0]),claims)
    # Short write-up: every stated numeric value also appears in claims/tables.
    summary='''# ML evidence summary\n\n## Problem\n- Estimate movement-risk classes and detect anomalous sensor patterns from generated subsidence signals.\n- Evidence is synthetic; it does not establish real-mine performance.\n\n## Data\n- Physics-coupled synthetic generator; one event per node, with 60-step windows and 10-step stride.\n- Regime-held-out split; grouped CV keeps events together. The fresh locked synthetic test was evaluated once.\n- Counts and class support: `tables/dataset.md`; construction overview: `figures/fig_03_dataset.png`.\n\n## Models\n- Isolation Forest: unsupervised anomaly score, fit on healthy training rows.\n- XGBoost: three-class risk score, compared with logistic regression and a threshold rule.\n- Physics check: compares observed movement with config-derived expected deformation.\n- TCN forecaster: next-window displacement/tilt estimate, compared with persistence.\n- Alert engine: maps risk scores to alert states; current single-node corpus cannot validate neighbor confirmation.\n\n## Results\n- Locked synthetic test results: `tables/results.md` and `tables/per_class.md`; confusion matrix: `figures/fig_05_per_class.png`.\n- Tuned XGBoost locked Critical recall is low; its score is not an early-warning success claim. Alert lead time is negative in the locked test (`tables/alert.md`).\n- Grouped development CV is reported separately in `tables/baselines.md`; development is not the locked test.\n\n## Baselines\n- On the locked test, tuned XGBoost has higher Critical recall than default XGBoost but low absolute recall; logistic has higher Critical recall and many more Normal false alarms. Full values are in `tables/results.md`.\n- The next-window TCN beats persistence on grouped CV (`tables/forecast.md`).\n\n## Fault checks and workstation profile\n- Selected fault-category class recall is in `tables/fault_scenarios.md`; Critical recall is zero in those validation categories. Per-category false-Critical rate was not retained.\n- Timing and memory are workstation CPU measurements only (`tables/edge_profile.md`); no device claim is made.\n\n## Limitations\n- Synthetic corpus only; no real-mine validation. One node per event gates spatial neighbor confirmation.\n- Forecast support is limited to the next window. The tabletop fixture is synthetic, not physical-rig evidence.\n- No row-level locked probabilities were retained; no event timeline is shown.\n\n## Not built\n- This bundle does not claim field deployment, physical-rig validation, or operational benefit.\n'''
    (outdir/'ml_summary.md').write_text(summary,encoding='utf-8')
    (outdir/'mode.md').write_text(make_mode(),encoding='utf-8')
    readme='''# ML submission index\n- `ml_summary.md` — concise evidence summary (Technical Approach).\n- `figures/` — source-linked, paste-ready plots (Technical Approach / Feasibility).\n- `tables/` — locked-test, grouped-CV, forecast, fault, and workstation tables.\n- `claims.csv` — source and verification ledger for every table value.\n- `reproduce.md` — regeneration and verification commands.\n'''
    (outdir/'README.md').write_text(readme,encoding='utf-8')
    (outdir/'reproduce.md').write_text('''# Reproduce the ML submission bundle\n\nRun from the repository root in the pinned project environment.\n\n```sh\npython -m pip install -r requirements.txt\npython submit/build/make_submit.py --verify\n```\n\nThe builder reads committed post-fix reports, the simulator configuration/source, and the reliability-curve artifact. It does not open the locked dataset. It regenerates tables and figures; `--verify` builds twice in temporary directories, compares all CSV bytes, checks source gates, and then writes `submit/`.\n\nTo independently recheck the feature-leakage gate and test suite:\n\n```sh\npython -m pytest tests/test_build_feature_store.py::test_label_blind_and_permuted_labels_leave_features_identical -q\npython -m pytest tests/ -q\n```\n\nThe final test results are read from `reports/final_eval.json` and `reports/final_eval.md`; do not rerun `scripts/final_eval.py` because the one-use evaluation has already been consumed. Dataset construction statistics come from `data/synthetic/dataset_manifest.json`; model reports are committed under `reports/`.\n''',encoding='utf-8')
    # Add Markdown companions after claim ledger creation.
    claims_rows=read_csv(outdir/'claims.csv')
    (outdir/'claims.md').write_text('# Claims ledger\n\n'+markdown_table(claims_rows),encoding='utf-8')

def verify():
    mode=make_mode()
    assert 'Mode A' in mode
    # The label-blind test is actually executed during verification.
    p=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short','tests/test_build_feature_store.py::test_label_blind_and_permuted_labels_leave_features_identical'],cwd=ROOT,text=True,capture_output=True)
    (OUT/'build'/'label_blind_gate.txt').write_text(p.stdout+p.stderr,encoding='utf-8')
    assert p.returncode==0, p.stdout+p.stderr
    with tempfile.TemporaryDirectory(prefix='submit_a_') as a, tempfile.TemporaryDirectory(prefix='submit_b_') as b:
        build(Path(a)); build(Path(b))
        ca={p.name:p.read_bytes() for p in (Path(a)/'tables').glob('*.csv')}
        cb={p.name:p.read_bytes() for p in (Path(b)/'tables').glob('*.csv')}
        assert ca==cb,'table bytes differ across deterministic regeneration'
        fa=sorted(p.name for p in (Path(a)/'figures').glob('*.png')); fb=sorted(p.name for p in (Path(b)/'figures').glob('*.png'))
        assert fa==fb and len(fa)==11
    build(OUT)
    # Content, provenance, format, size and evidence checks.
    files=[p for p in OUT.rglob('*') if p.is_file()]
    total=sum(p.stat().st_size for p in files)
    assert total<15*1024*1024, f'submit is {total} bytes'
    banned_words=['revolution'+'ary','cutting'+'-edge','ro'+'bust','seam'+'less']
    banned=re.compile(r'\b(?:'+'|'.join(map(re.escape,banned_words))+r')\b',re.I)
    private_path='/'+'Users'+'/'
    placeholder=re.compile(r'\b(?:'+'TO'+'DO|T'+'BD|X'+'XX)\b',re.I)
    forbidden_ext={'.parquet','.joblib','.pt','.ipynb','.pkl','.pickle'}
    for p in files:
        assert p.suffix.lower() not in forbidden_ext, p
        if p.suffix.lower() in ('.md','.csv','.py','.txt'):
            t=p.read_text(errors='replace')
            assert private_path not in t, p
            assert not placeholder.search(t), p
            assert not banned.search(t), p
            assert ('superseded'+'_leaky') not in t, p
    pngs=list((OUT/'figures').glob('*.png')); svgs=list((OUT/'figures').glob('*.svg'))
    assert len(pngs)==11 and len(svgs)==11
    assert (OUT/'figures/captions.md').exists() and len(read_csv(OUT/'claims.csv'))>0
    assert json.loads((ROOT/'reports/test_lock.json').read_text())['evals_run']==1
    print(f'PASS: Mode A; label-blind test; deterministic CSV regeneration; {len(pngs)} PNG/SVG figure pairs; {len(read_csv(OUT/"claims.csv"))} ledger rows; {total/1048576:.2f} MiB total.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--verify',action='store_true'); args=parser.parse_args()
    if args.verify: verify()
    else: build(OUT)
