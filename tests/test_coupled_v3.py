import dataclasses
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from src.simulator.coupled_v3 import Parameters,ScenarioField,simulate,severity,generate_corpus
from src.features.causal_v3 import build_features,attach_targets
from src.features.windowing import _velocity,_velocity_v2

CFG=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/generalization_v3.yaml').read_text())


def params(**kwargs):
    p=Parameters('rapid_subsidence','exponential',20.,30.,60.,2100.,2.,16.,2.,1.6,0.,0.,0.,0.,'NONE')
    return dataclasses.replace(p,**kwargs)


@pytest.mark.parametrize('scenario',['rapid_subsidence','accelerating_subsidence','irregular_subsidence','multiple_zones','stable_ground'])
@pytest.mark.parametrize('shape',['exponential','power','ramp','sigmoid','smoothstep','two_stage'])
def test_gradients_match_displacement_field(scenario,shape):
    f=ScenarioField(params(scenario=scenario,shape=shape));t=np.arange(24.);eps=1e-4
    gx,gy=f.gradient(20.,30.,t)
    np.testing.assert_allclose(gx,(f.value(20+eps,30,t)-f.value(20-eps,30,t))/(2*eps),atol=1e-7)
    np.testing.assert_allclose(gy,(f.value(20,30+eps,t)-f.value(20,30-eps,t))/(2*eps),atol=1e-7)


def test_sensor_channels_follow_same_field_and_strain_geometry():
    p=params();f=ScenarioField(p);r,l=simulate(p,np.random.default_rng(2),CFG,'e','one_trapdoor')
    t=r.timestamp.to_numpy();gx,gy=f.gradient(p.x,p.y,t)
    np.testing.assert_allclose(r.displacement,f.value(p.x,p.y,t))
    np.testing.assert_allclose(np.tan(np.radians(r.tilt_x))*1000,gx)
    np.testing.assert_allclose(np.tan(np.radians(r.tilt_y))*1000,gy)
    ux,uy=f.horizontal(p.x,p.y,t);vx,vy=f.horizontal(275.,275.,t)
    distance=np.hypot(275+vx-p.x-ux,275+vy-p.y-uy);d0=np.hypot(275-p.x,275-p.y)
    np.testing.assert_allclose(r.strain,(distance-d0)/d0)
    assert (l.loc[t<=p.onset,'risk_label']=='NORMAL').all()
    assert (l.legacy_risk_label=='CRITICAL').all()


def test_no_spurious_stable_ground_movement():
    r,_=simulate(params(scenario='stable_ground'),np.random.default_rng(2),CFG,'e','n')
    assert np.max(np.abs(r[['displacement','tilt_x','tilt_y','strain']].to_numpy()))==0


def test_disturbance_and_vibration_have_observable_signals():
    for scenario in ['single_node_disturbance','vibration_only']:
        r,l=simulate(params(scenario=scenario),np.random.default_rng(2),CFG,'e','n')
        active=l.anomaly_label==1
        assert active.any()
        if scenario=='vibration_only':assert r.loc[active,'vibration_rms'].mean()>5*r.loc[~active,'vibration_rms'].mean()
        else:assert r.loc[active,'displacement'].max()>5
        assert (l.risk_label=='NORMAL').all() # no displacement severity from vibration alone


def test_corrected_velocity_uses_finite_endpoint_time_span():
    v=np.array([np.nan,2.,4.,6.,np.nan])
    assert _velocity(v,.5)==pytest.approx(4.)
    assert _velocity_v2(v,.5)==pytest.approx(2.)


def test_v3_features_are_label_blind_and_causal():
    r,l=simulate(params(),np.random.default_rng(2),CFG,'e','n');f=build_features(r)
    polluted=r.copy()
    for c in ['risk_label','true_displacement_mm','scenario_family','generation_parameter_id']:polluted[c]='secret'
    pd.testing.assert_frame_equal(f,build_features(polluted))
    short=build_features(r.iloc[:100]);pd.testing.assert_frame_equal(f.iloc[:len(short)].reset_index(drop=True),short)
    targets=attach_targets(f,l)
    expected=l.set_index('timestamp').loc[f.window_timestamp,'risk_label'].to_numpy()
    np.testing.assert_array_equal(targets.risk_label,expected)


def test_policy_thresholds_preserve_existing_strict_boundaries():
    assert list(severity([0,15,15.01,35,35.01],CFG['label_policy']))==['NORMAL','NORMAL','WARNING','WARNING','CRITICAL']


def test_noise_replicates_share_parameter_groups_and_generation_reproducible():
    from copy import deepcopy
    cfg=deepcopy(CFG);cfg['simulation']['scenarios']=['stable_ground','rapid_subsidence'];cfg['development']['groups_per_scenario']=2
    a,b,p=generate_corpus(cfg);c,d,q=generate_corpus(cfg)
    pd.testing.assert_frame_equal(a,c);pd.testing.assert_frame_equal(b,d);pd.testing.assert_frame_equal(p,q)
    assert b.groupby('generation_parameter_id').event_id.nunique().eq(2).all()


def test_drift_uses_hours_to_days_cadence_not_one_day_per_sample():
    p=params(scenario='sensor_drift',sensor_fault='DRIFT')
    r,_=simulate(p,np.random.default_rng(2),CFG,'e','n')
    onset=int(.35*CFG['simulation']['steps'])
    from src.config import physics_config
    expected=physics_config()['faults']['drift_mm_per_day']*CFG['simulation']['interval_hours']/24*(len(r)-1-onset)
    assert r.displacement.iloc[-1]==pytest.approx(expected)
    assert expected<2


def test_temporal_probabilities_are_causal_and_reset_per_event():
    from src.risk.severity_v3 import SeverityBundle
    class Estimator:
        classes_=np.array([0,1,2])
        def predict_proba(self,x):return x.copy()
    f=pd.DataFrame({'event_id':['a','a','b'],'node_id':['n']*3,'window_timestamp':[0.,1.,0.],
                    'p0':[.8,.1,.1],'p1':[.1,.2,.2],'p2':[.1,.7,.7]})
    m=SeverityBundle(Estimator(),['p0','p1','p2'],None,{'warning':.5,'critical':.5},.7)
    result=m.predict_proba(f)
    np.testing.assert_allclose(result[0],[.8,.1,.1])
    np.testing.assert_allclose(result[2],[.1,.2,.7])
    np.testing.assert_allclose(result[:2],m.predict_proba(f.iloc[:2]))


def test_no_label_or_hidden_parameter_enters_v3_feature_registry():
    from src.features.causal_v3 import schema
    cfg=schema()
    assert set(cfg['features']).isdisjoint(cfg['excluded'])
    assert set(cfg['provenance'])==set(cfg['features'])


def test_legacy_scenario_identity_can_conflict_for_identical_observations():
    # This is a label-identifiability audit, not an extra training scenario.
    a,la=simulate(params(scenario='rapid_subsidence',shape='power'),np.random.default_rng(5),CFG,'same','node')
    b,lb=simulate(params(scenario='accelerating_subsidence',shape='power'),np.random.default_rng(5),CFG,'same','node')
    pd.testing.assert_frame_equal(a,b)
    np.testing.assert_array_equal(la.risk_label,lb.risk_label)
    assert (la.legacy_risk_label=='CRITICAL').all()
    assert (lb.legacy_risk_label=='WARNING').all()
