"""Frozen optimization inference contract; no label reads or runtime fitting."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

CLASSES = ('NORMAL', 'WARNING', 'CRITICAL')


def threshold_predictions(prob, thresholds):
    """Strict gates: Critical takes precedence; argmax cannot bypass a gate."""
    prob = np.asarray(prob)
    if prob.ndim != 2 or prob.shape[1] != 3 or not np.isfinite(prob).all():
        raise ValueError('expected finite N x 3 probabilities')
    if (prob < 0).any() or (prob > 1).any() or not np.allclose(prob.sum(1), 1, atol=1e-5):
        raise ValueError('invalid probability simplex')
    pred = np.full(len(prob), 'NORMAL', dtype='<U8')
    pred[prob[:, 1] >= thresholds['warning']] = 'WARNING'
    pred[prob[:, 2] >= thresholds['critical']] = 'CRITICAL'
    return pred


def apply_calibration(prob, calibration):
    if calibration is None:
        return prob
    columns = []
    for i, label in enumerate(CLASSES):
        model = calibration['models'][label]
        if calibration['method'] == 'sigmoid':
            columns.append(model.predict_proba(prob[:, i, None])[:, 1])
        else:
            columns.append(model.predict(prob[:, i]))
    out = np.clip(np.column_stack(columns), 1e-8, 1)
    return out / out.sum(axis=1, keepdims=True)


@dataclass
class OptimizedRiskModel:
    model: object
    features: list
    calibration: object
    thresholds: dict

    def predict_proba(self, frame: pd.DataFrame):
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError(f'missing features: {sorted(missing)}')
        if list(self.model.classes_) != [0, 1, 2]:
            raise ValueError('model class order must be NORMAL, WARNING, CRITICAL')
        x = frame[self.features].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError('nonfinite features: preprocessing/schema mismatch')
        return apply_calibration(self.model.predict_proba(x), self.calibration)

    def predict(self, frame):
        return threshold_predictions(self.predict_proba(frame), self.thresholds)
