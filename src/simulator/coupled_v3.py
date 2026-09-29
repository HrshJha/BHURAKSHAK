"""Versioned, single-stream synthetic observations from a shared deformation field.

The 15/35 mm label policy is inherited from the existing tabletop protocol.
It is a local severity target, NOT the legacy scenario-identity target and
NOT a validated mine hazard boundary. Hidden truth stays in a separate table.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np
import pandas as pd
from src.config import physics_config
from src.simulator.scenarios import Scenario, scenario_risk_mapping
from src.simulator.grid import build_grid
from src.simulator.faults import inject_fault, FaultType


@dataclass(frozen=True)
class Parameters:
    scenario: str
    shape: str
    x: float
    y: float
    sigma: float
    amplitude: float
    onset: float
    duration: float
    rate: float
    exponent: float
    displacement_noise: float
    tilt_noise: float
    strain_noise: float
    drift: float
    sensor_fault: str

    @property
    def group_id(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


SUBSIDENCE={'slow_subsidence','rapid_subsidence','accelerating_subsidence','irregular_subsidence','multiple_zones'}
FAULTS={'sensor_bias':'BIAS','sensor_stuck':'STUCK','sensor_dropout':'DROPOUT','sensor_spike':'SPIKE','sensor_drift':'DRIFT'}


def temporal(p,t):
    """Bounded smooth histories, zero before onset; units mm and hours."""
    age=np.maximum(np.asarray(t)-p.onset,0.)
    u=np.clip(age/p.duration,0,1)
    if p.shape=='exponential':q=1-np.exp(-p.rate*age/24)
    elif p.shape=='power':q=u**p.exponent
    elif p.shape=='ramp':q=u
    elif p.shape=='smoothstep':q=3*u*u-2*u*u*u
    elif p.shape=='sigmoid':
        logistic=lambda z:1/(1+np.exp(-10*(z-.5)))
        q=(logistic(u)-logistic(0))/(logistic(1)-logistic(0))
    elif p.shape=='two_stage':q=.35*np.clip(u/.3,0,1)+.65*np.clip((u-.55)/.45,0,1)**2
    else:raise ValueError(f'unknown temporal shape: {p.shape}')
    return p.amplitude*q


class ScenarioField:
    """One field supplies displacement, analytic gradient and convergence."""
    def __init__(self,p):self.p=p
    def components(self,x,y,t):
        p=self.p;x=np.asarray(x);y=np.asarray(y);t=np.asarray(t)
        if p.scenario not in SUBSIDENCE:
            return [(np.zeros(np.broadcast_shapes(x.shape,y.shape,t.shape)),x,y)]
        centers=[(0.,0.,1.)]
        if p.scenario=='multiple_zones':centers.append((float(physics_config()['scenarios']['second_zone_offset_m']),0.,.65))
        result=[]
        for cx,cy,weight in centers:
            dx=x-cx;dy=y-cy;k=np.exp(-(dx*dx+dy*dy)/(2*p.sigma**2))
            value=temporal(p,t)*weight*k
            if p.scenario=='irregular_subsidence':
                # Shared spatial field modulated by a small smooth temporal oscillation.
                age=np.maximum(t-p.onset,0);value*=1+.12*np.sin(2*np.pi*age/p.duration)
            result.append((value,dx,dy))
        return result
    def value(self,x,y,t):return sum(v for v,_,_ in self.components(x,y,t))
    def gradient(self,x,y,t):
        parts=self.components(x,y,t);return (sum(-dx*v/self.p.sigma**2 for v,dx,_ in parts),sum(-dy*v/self.p.sigma**2 for v,_,dy in parts))
    def horizontal(self,x,y,t):
        factor=float(physics_config()['strain']['horizontal_displacement_factor'])
        parts=self.components(x,y,t)
        return (sum(-factor*(v/1000)*dx/self.p.sigma for v,dx,_ in parts),sum(-factor*(v/1000)*dy/self.p.sigma for v,_,dy in parts))
    def strain(self,x,y,rx,ry,t):
        ux,uy=self.horizontal(x,y,t);vx,vy=self.horizontal(rx,ry,t)
        d0=np.hypot(rx-x,ry-y)
        if d0<=0:raise ValueError('strain reference is co-located')
        return (np.hypot(rx+vx-x-ux,ry+vy-y-uy)-d0)/d0


def severity(displacement,policy):
    v=np.asarray(displacement)
    return np.where(v>policy['critical_mm'],'CRITICAL',np.where(v>policy['warning_mm'],'WARNING','NORMAL'))


def simulate(p,rng,cfg,event_id,node_id):
    spec=cfg['simulation'];t=np.arange(spec['steps'])*spec['interval_hours'];field=ScenarioField(p)
    truth=field.value(p.x,p.y,t);gx,gy=field.gradient(p.x,p.y,t)
    # atan of the gradient in m/m gives physical tilt in degrees.
    tx=np.degrees(np.arctan(gx/1000))+rng.normal(0,p.tilt_noise,len(t))
    ty=np.degrees(np.arctan(gy/1000))+rng.normal(0,p.tilt_noise,len(t))
    strain=field.strain(p.x,p.y,275.,275.,t)+rng.normal(0,p.strain_noise,len(t))
    disp=truth+rng.normal(0,p.displacement_noise,len(t))+p.drift*t/24
    fault=np.full(len(t),'NONE',object)
    if p.sensor_fault!='NONE':
        outcome=inject_fault(disp,FaultType(p.sensor_fault),rng,int(.35*len(t)),days_per_step=spec['interval_hours']/24);disp=outcome.values;fault[outcome.fault_mask]=p.sensor_fault
    # RMS/peak summaries of an on-node background plus envelope-modulated vibration.
    base=float(physics_config()['noise']['vibration_noise_std'])
    samples=rng.normal(0,base,(len(t),128));seconds=np.arange(128)/100.
    anomaly=(truth>float(physics_config()['scenarios']['anomaly_subsidence_threshold_mm'])) | (fault!='NONE')
    if p.scenario=='vibration_only':
        active=(t>=p.onset)&(t<p.onset+1.)
        samples+=active[:,None]*float(physics_config()['vibration']['burst_amplitude'])*np.sin(2*np.pi*40*seconds)
        anomaly|=active
    if p.scenario=='single_node_disturbance':
        width=.4;center=p.onset+1.;pulse=float(physics_config()['scenarios']['local_anomaly_amplitude_mm'])*np.exp(-.5*((t-center)/width)**2)
        disp+=pulse;anomaly|=pulse>1.;samples+=(pulse/8)[:,None]*.08*np.sin(2*np.pi*12*seconds)
    if p.scenario=='slow_drift_temperature':disp+=1.2*np.sin(2*np.pi*t/24)
    # Weak vibration is a supporting channel linked to local motion, never a class code.
    speed=np.gradient(truth,t);envelope=.01*np.sqrt(np.maximum(np.abs(speed),0))
    samples+=envelope[:,None]*rng.normal(0,1,samples.shape)
    vibration_rms=np.sqrt(np.mean(samples**2,axis=1));vibration_peak=np.max(np.abs(samples),axis=1)
    lost=np.zeros(len(t),bool)
    if p.scenario=='packet_loss':lost=rng.random(len(t))<.05
    if p.scenario=='communication_failure':lost=t>=t.max()*.8
    disp[lost]=np.nan;tx[lost]=np.nan;ty[lost]=np.nan;strain[lost]=np.nan
    raw=pd.DataFrame(dict(event_id=event_id,node_id=node_id,timestamp=t,x=p.x,y=p.y,displacement=disp,tilt_x=tx,tilt_y=ty,
        tilt_magnitude=np.hypot(tx,ty),strain=strain,vibration_rms=vibration_rms,vibration_peak=vibration_peak,
        battery=4.1-.5*t/24+rng.normal(0,.01,len(t)),RSSI=rng.normal(-95,6,len(t)),SNR=rng.normal(0,5,len(t)),packet_loss=lost.astype(int)))
    scenario_label=scenario_risk_mapping()[p.scenario]
    if p.scenario=='multiple_zones':scenario_label='CRITICAL' if np.hypot(p.x,p.y)<=np.hypot(p.x-130,p.y) else 'WARNING'
    labels=pd.DataFrame(dict(event_id=event_id,node_id=node_id,timestamp=t,risk_label=severity(truth,cfg['label_policy']),
        legacy_risk_label=scenario_label,anomaly_label=anomaly.astype(int),fault_label=fault,true_displacement_mm=truth,true_velocity_mm_per_hour=speed,
        generation_parameter_id=p.group_id,scenario_family=p.scenario,shape_family=p.shape))
    return raw,labels


def generate_corpus(cfg,section='development'):
    spec=cfg[section];sim=cfg['simulation'];rng=np.random.default_rng(spec.get('seed',cfg['seed']));grid=build_grid()
    raws=[];labels=[];params=[]
    def draw(key):return float(rng.uniform(*sim[key]))
    for scenario in sim['scenarios']:
        for group in range(spec['groups_per_scenario']):
            node=int(rng.integers(grid.n_nodes));noise_mult=float(spec.get('noise_multiplier',1.))
            fault=FAULTS.get(scenario,'NONE')
            if scenario in SUBSIDENCE and rng.random()<sim['subsidence_fault_probability']:fault=str(rng.choice(list(FAULTS.values())))
            rate=draw('rate_per_day')
            # Scenario names retain distinct rate tendencies; severity is still local truth.
            if scenario=='rapid_subsidence':rate=max(rate,sim['rapid_min_rate_per_day'])
            if scenario=='slow_subsidence':rate=min(rate,sim['slow_max_rate_per_day'])
            p=Parameters(scenario,str(rng.choice(spec['shapes'])),float(grid.x[node]),float(grid.y[node]),draw('sigma_m'),
               2100*draw('amplitude_scale'),draw('onset_hours'),draw('duration_hours'),rate,float(rng.uniform(1.2,2.4)),
               draw('displacement_noise_mm')*noise_mult,draw('tilt_noise_deg')*noise_mult,draw('strain_noise')*noise_mult,draw('drift_mm_per_day'),fault)
            params.append(dict(generation_parameter_id=p.group_id,**asdict(p)))
            for repeat in range(sim['repetitions_per_parameter_group']):
                eid=f'V3_{section}_{scenario}_{group:04}_{repeat}'
                a,b=simulate(p,rng,cfg,eid,grid.node_ids[node]);raws.append(a);labels.append(b)
    return pd.concat(raws,ignore_index=True),pd.concat(labels,ignore_index=True),pd.DataFrame(params)
