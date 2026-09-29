"""Version 3 observable-only causal features. No truth/labels reach emitters."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.features.windowing import _velocity

KEYS=['event_id','node_id','window_index','window_timestamp']


def schema():
    return yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/feature_schema_v3.yaml').read_text())


def build_features(raw):
    """One row per completed window; current and past samples only."""
    cfg=schema();window=cfg['window_steps'];stride=cfg['stride_steps'];short=cfg['short_steps'];end_steps=cfg['endpoint_steps']
    channels=cfg['observable_channels'];required={'event_id','node_id','timestamp',*channels}
    if not required<=set(raw):raise ValueError(f'missing observations: {sorted(required-set(raw))}')
    # Deliberate whitelist prevents annotations from ever becoming inputs.
    obs=raw[['event_id','node_id','timestamp',*channels]]
    rows=[]
    for (eid,nid),g in obs.groupby(['event_id','node_id'],sort=False):
        g=g.sort_values('timestamp');t=g.timestamp.to_numpy(float)
        if np.any(np.diff(t)<=0):raise ValueError('timestamps must be strictly increasing')
        for start in range(0,len(g)-window+1,stride):
            end=start+window;part=g.iloc[start:end];row=dict(event_id=eid,node_id=nid,window_index=start//stride,window_timestamp=t[end-1])
            for ch in ['displacement','tilt_x','tilt_y','strain']:
                v=part[ch].to_numpy(float);recent=v[-short:];finite=np.isfinite(recent)
                tail=v[-end_steps:];tail=tail[np.isfinite(tail)]
                row[ch+'_endpoint']=float(np.median(tail)) if len(tail) else np.nan
                row[ch+'_mean']=float(np.nanmean(recent)) if finite.any() else np.nan
                row[ch+'_std']=float(np.nanstd(recent)) if finite.any() else np.nan
                row[ch+'_velocity']=_velocity(recent,float(np.median(np.diff(t[end-short:end]))))
                # Robustly recover current value/trend from a local quadratic; no future samples.
                if finite.sum()>=4:
                    tau=t[end-short:end]-t[end-1];coef=np.polyfit(tau[finite],recent[finite],2)
                    row[ch+'_projected']=float(coef[2]);row[ch+'_acceleration']=float(2*coef[0])
                else:row[ch+'_projected']=np.nan;row[ch+'_acceleration']=np.nan
            initial=part.displacement.iloc[:end_steps].dropna()
            row['displacement_change']=row['displacement_endpoint']-(float(initial.median()) if len(initial) else np.nan)
            row['displacement_missing_ratio']=float(part.displacement.isna().mean())
            row['channel_missing_ratio']=float(part[['displacement','tilt_x','tilt_y','strain']].isna().to_numpy().mean())
            for ch in ['vibration_rms','vibration_peak','packet_loss']:
                row[ch]=float(part[ch].iloc[-short:].mean())
            rows.append(row)
    out=pd.DataFrame(rows)
    if set(out.columns)!=set(KEYS+cfg['features']):raise ValueError('v3 feature schema mismatch')
    return out[KEYS+cfg['features']]


def attach_targets(features,truth):
    """Target assembly is separate from feature computation; endpoint alignment."""
    target=truth.rename(columns={'timestamp':'window_timestamp'})
    return features.merge(target,on=['event_id','node_id','window_timestamp'],validate='one_to_one',how='left')
