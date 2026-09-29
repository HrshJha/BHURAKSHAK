"""Evaluation boundary regressions, exercised without the locked corpus."""
import numpy as np
import pandas as pd
import pytest
from src.risk.optimized import threshold_predictions, OptimizedRiskModel
from scripts.optimize_bhurakshak import inner_parts, folds, select_thresholds
from scripts.evaluate_optimization import claim
from scripts.tune_models import grouped_folds


def frame():
    rows=[]
    for group in range(120):
        label=['NORMAL','WARNING','CRITICAL'][group%3]
        for event in range(2):
            for window in range(3):
                rows.append(dict(generation_parameter_id=f'g{group}',event_id=f'e{group}-{event}',risk_label=label,window_index=window))
    return pd.DataFrame(rows)


def test_thresholds_cannot_be_bypassed_by_argmax():
    p=np.array([[.15,.35,.5],[.1,.7,.2],[.05,.15,.8]])
    assert list(threshold_predictions(p,{'warning':.75,'critical':.6}))==['NORMAL','NORMAL','CRITICAL']


def test_threshold_selection_enforces_joint_normal_fpr():
    f=frame();rng=np.random.default_rng(11);p=rng.dirichlet([2,1,1],len(f))
    for cap in [.01,.05,.1,.2]:
        t=select_thresholds(f,p,cap);pred=threshold_predictions(p,t)
        assert np.mean(pred[f.risk_label=='NORMAL']!='NORMAL')<=cap+1e-12


def test_inner_and_outer_parameter_groups_disjoint():
    f=frame()
    for parts,validation in folds(f):
        assert sum(map(len,parts))+len(validation)==len(f)
        for i,p in enumerate(parts):
            assert not set(p.generation_parameter_id)&set(validation.generation_parameter_id)
            for other in parts[:i]:assert not set(p.generation_parameter_id)&set(other.generation_parameter_id)


def test_legacy_tuner_groups_shared_parameter_events():
    f=frame()
    for a,b in grouped_folds(f):
        assert not set(f[f.event_id.isin(a)].generation_parameter_id)&set(f[f.event_id.isin(b)].generation_parameter_id)


def test_test_claim_is_one_use(tmp_path):
    path=tmp_path/'lock.json';claim(path,{'evals_run':1})
    with pytest.raises(FileExistsError):claim(path,{'evals_run':1})


def test_invalid_probabilities_fail_closed():
    with pytest.raises(ValueError):threshold_predictions(np.array([[.4,.4,.4]]),{'critical':.5,'warning':.5})


def test_inference_checks_schema_before_model_execution():
    model=OptimizedRiskModel(None,['velocity'],None,{})
    with pytest.raises(ValueError,match='missing features'):model.predict_proba(pd.DataFrame({'wrong':[1]}))


def test_requested_gated_features_do_not_silently_enable_all_inputs():
    from src.risk.xgboost_model import train_risk_model, XGBoostModelError
    f=pd.DataFrame({'displacement':[0.], 'split':['train'], 'risk_label':['NORMAL']})
    with pytest.raises(XGBoostModelError,match='requested features'):
        train_risk_model(f,feature_groups=['hotspot_density'])


def test_inference_feature_order_and_label_blindness():
    class Estimator:
        classes_=np.array([0,1,2])
        def predict_proba(self,x):
            np.testing.assert_array_equal(x,np.array([[1.,2.]]))
            return np.array([[.1,.2,.7]])
    model=OptimizedRiskModel(Estimator(),['a','b'],None,{'warning':.6,'critical':.6})
    f=pd.DataFrame({'b':[2.],'risk_label':['NORMAL'],'a':[1.]})
    assert model.predict(f)[0]=='CRITICAL'
    f['risk_label']='CRITICAL'
    assert model.predict(f)[0]=='CRITICAL'
