""" comparison baselines.

/: XGBoost must be evaluated against threshold-rule, logistic-regression
and random-forest baselines — all trained and predicting on the **same feature
matrix** as XGBoost. If ML does not beat the explainable threshold rule,
"that's a sign something is wrong in the pipeline, not that ML isn't needed".

The threshold rule is the -style explainable policy: sustained displacement
rate / acceleration / level over ``persistence_windows`` consecutive windows.
Its thresholds come from configs/risk_model.yaml and are calibrated on
the validation split only by:func:`calibrate_threshold_rule`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import risk_model_config
from src.risk.xgboost_model import XGBoostModelError

__all__ = ["BaselinesError", "fit_baselines", "threshold_rule_predict", "calibrate_threshold_rule"]

CLASSES = ["NORMAL", "WARNING", "CRITICAL"]


class BaselinesError(ValueError):
    """Raised on invalid baseline inputs."""


@dataclass
class Baselines:
    logistic: LogisticRegression
    logistic_scaler: StandardScaler
    random_forest: RandomForestClassifier
    features: list[str]
    classes: list[str]

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise BaselinesError(f"feature columns missing: {missing}")
        return df[self.features].to_numpy(dtype=float)

    def predict_logistic(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.logistic.predict(self.logistic_scaler.transform(self._matrix(df)))
        return np.asarray(pred, dtype=object)

    def predict_random_forest(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.random_forest.predict(self._matrix(df))
        return np.asarray(pred, dtype=object)


def _classes_from(df: pd.DataFrame) -> list[str]:
    classes = sorted(df["risk_label"].unique())
    if set(classes) != set(CLASSES):
        raise BaselinesError(f"baselines expect the 3-class MVP classes {sorted(CLASSES)}, got {classes}")
    return classes


def fit_baselines(train: pd.DataFrame, features: list[str]) -> Baselines:
    """Fit logistic + random-forest baselines on the SAME feature matrix as XGBoost.

 ``train`` must be the TRAIN split only (the caller owns split discipline).
 """
    classes = _classes_from(train)
    missing = [c for c in features if c not in train.columns]
    if missing:
        raise BaselinesError(f"feature columns missing: {missing}")
    X = train[features].to_numpy(dtype=float)
    y = train["risk_label"].to_numpy()

    cfg = risk_model_config()["baselines"]
    scaler = StandardScaler().fit(X)
    logistic = LogisticRegression(
        C=float(cfg["logistic"]["C"]),
        max_iter=int(cfg["logistic"]["max_iter"]),
        random_state=42,
    )
    logistic.fit(scaler.transform(X), y)

    rf_cfg = cfg["random_forest"]
    rf = RandomForestClassifier(
        n_estimators=int(rf_cfg["n_estimators"]),
        random_state=int(rf_cfg["random_state"]),
        class_weight=rf_cfg["class_weight"],
        n_jobs=-1,
    )
    rf.fit(X, y)
    return Baselines(
        logistic=logistic,
        logistic_scaler=scaler,
        random_forest=rf,
        features=features,
        classes=classes,
    )


def threshold_rule_predict(
    df: pd.DataFrame,
    *,
    velocity_warning: float | None = None,
    accel_critical: float | None = None,
    score_warning: float | None = None,
    persistence_windows: int | None = None,
) -> np.ndarray:
    """/ threshold rule on the SAME feature matrix (explainable baseline).

 A window is a candidate WARNING if its displacement ``velocity`` exceeds
 ``velocity_warning`` OR its level ``displacement`` exceeds ``score_warning``;
 it becomes CRITICAL if additionally ``acceleration`` exceeds
 ``accel_critical``. A label sticks only when the condition held for
 ``persistence_windows`` consecutive windows within the same (event, node)
 series — mirroring the persistence philosophy.
 """
    cfg = risk_model_config()["baselines"]["threshold_rule"]
    velocity_warning = float(velocity_warning if velocity_warning is not None else cfg["velocity_warning"])
    accel_critical = float(accel_critical if accel_critical is not None else cfg["accel_critical"])
    score_warning = float(score_warning if score_warning is not None else cfg["score_warning"])
    persistence = int(persistence_windows if persistence_windows is not None else cfg["persistence_windows"])

    needed = {"displacement", "velocity", "acceleration"}
    missing = needed - set(df.columns)
    if missing:
        raise BaselinesError(f"threshold rule needs columns {sorted(missing)}")

    out = pd.Series("NORMAL", index=df.index, dtype=object)
    for _key, g in df.groupby(["event_id", "node_id"], sort=False):
        idx = g.index
        level = g["displacement"].to_numpy(dtype=float)
        vel = g["velocity"].to_numpy(dtype=float)
        acc = g["acceleration"].to_numpy(dtype=float)
        warn = (vel > velocity_warning) | (level > score_warning)
        crit = warn & (acc > accel_critical)
        # persistence: label only if the condition held for `persistence` consecutive windows
        def sustained(mask: np.ndarray) -> np.ndarray:
            run = 0
            keep = np.zeros(mask.size, dtype=bool)
            for i, m in enumerate(mask):
                run = run + 1 if m else 0
                keep[i] = run >= persistence
            return keep
        pred = np.where(sustained(crit), "CRITICAL", np.where(sustained(warn), "WARNING", "NORMAL"))
        out.loc[idx] = pred
    return out.to_numpy()


def calibrate_threshold_rule(val: pd.DataFrame) -> dict:
    """Tune the rule's thresholds on the VALIDATION split for best macro-F1.

 A small grid around the configured starting points; the winning values are
 returned so the test evaluation uses validation-chosen parameters only.
 """
    from sklearn.metrics import f1_score

    cfg = risk_model_config()["baselines"]["threshold_rule"]
    best: dict | None = None
    for v_mult in (1 / 2, 1.0, 2.0):
        for a_mult in (1 / 2, 1.0, 2.0):
            for s_mult in (1 / 2, 1.0, 2.0):
                params = dict(
                    velocity_warning=float(cfg["velocity_warning"]) * v_mult,
                    accel_critical=float(cfg["accel_critical"]) * a_mult,
                    score_warning=float(cfg["score_warning"]) * s_mult,
                    persistence_windows=int(cfg["persistence_windows"]),
                )
                pred = threshold_rule_predict(val, **params)
                score = float(f1_score(val["risk_label"], pred, average="macro", zero_division=0))
                if best is None or score > best["macro_f1"]:
                    best = {"macro_f1": score, "params": params}
    if best is None:
        raise BaselinesError("threshold-rule calibration failed")
    return best
