"""Versioned local-severity inference, with fitted preprocessing and causal smoothing."""
from dataclasses import dataclass
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from xgboost import XGBClassifier
from src.risk.optimized import CLASSES,apply_calibration,threshold_predictions


class TwoStageClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,n_estimators=250,max_depth=4,learning_rate=.08,seed=1731):
        self.n_estimators=n_estimators;self.max_depth=max_depth;self.learning_rate=learning_rate;self.seed=seed
    def fit(self,x,y,sample_weight=None):
        config=dict(n_estimators=self.n_estimators,max_depth=self.max_depth,learning_rate=self.learning_rate,tree_method='hist',n_jobs=1,random_state=self.seed,objective='binary:logistic',eval_metric='logloss')
        self.critical_=XGBClassifier(**config).fit(x,(y==2).astype(int),sample_weight=sample_weight)
        non=y!=2
        self.warning_=XGBClassifier(**config).fit(x[non],(y[non]==1).astype(int),sample_weight=None if sample_weight is None else sample_weight[non])
        self.classes_=np.array([0,1,2]);self.n_features_in_=x.shape[1];return self
    def predict_proba(self,x):
        critical=self.critical_.predict_proba(x)[:,1];warning=self.warning_.predict_proba(x)[:,1]
        return np.column_stack([(1-critical)*(1-warning),(1-critical)*warning,critical])
    def predict(self,x):return self.predict_proba(x).argmax(1)


@dataclass
class SeverityBundle:
    model: object
    features: list
    calibration: object
    thresholds: dict
    alpha: float=1.
    feature_version: str='v3'
    target_policy: str='local_displacement_severity_v1'

    def predict_proba(self,frame):
        missing=set(self.features)-set(frame)
        if missing:raise ValueError(f'missing v3 features: {sorted(missing)}')
        x=frame[self.features].to_numpy(float)
        if np.isinf(x).any():raise ValueError('infinite observation; refuse inference')
        if list(self.model.classes_)!=[0,1,2]:raise ValueError('class order mismatch')
        p=apply_calibration(self.model.predict_proba(x),self.calibration)
        if self.alpha<1:
            if not {'event_id','node_id','window_timestamp'}<=set(frame):raise ValueError('temporal inference requires series/time keys')
            work=frame[['event_id','node_id','window_timestamp']].copy();work['_position']=np.arange(len(work))
            for _,g in work.groupby(['event_id','node_id'],sort=False):
                positions=g.sort_values('window_timestamp')._position.to_numpy()
                for last,current in zip(positions[:-1],positions[1:]):p[current]=self.alpha*p[current]+(1-self.alpha)*p[last]
        return p

    def predict(self,frame):return threshold_predictions(self.predict_proba(frame),self.thresholds)
