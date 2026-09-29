""" acceptance tests — comparison baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.baselines import (
    BaselinesError,
    calibrate_threshold_rule,
    fit_baselines,
    threshold_rule_predict,
)
from src.risk.xgboost_model import train_risk_model

FEATURES = ["displacement", "velocity", "acceleration", "tilt_x", "physics_residual", "anomaly_score"]


def _frame(n: int = 2400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n_per = n // 3
    for cls, (mu_d, mu_v, mu_a) in enumerate(((2.0, 0.005, 0.0005), (18.0, 0.08, 0.01), (42.0, 0.25, 0.03))):
        d = rng.normal(mu_d, 2.0, n_per)
        v = rng.normal(mu_v, 0.01, n_per)
        rows.append(
            pd.DataFrame(
                {
                    "event_id": [f"E_{cls}_{i // 20:02d}_000" for i in range(n_per)]
                }
            )
        )
        rows[-1]["node_id"] = [f"V{i % 4:04d}" for i in range(n_per)]
        rows[-1]["displacement"] = d
        rows[-1]["velocity"] = v
        rows[-1]["acceleration"] = np.abs(rng.normal(mu_a, mu_a / 2, n_per)) + 1e-4
        rows[-1]["tilt_x"] = d / 400.0
        rows[-1]["physics_residual"] = rng.normal(0, 1.0, n_per)
        rows[-1]["anomaly_score"] = np.clip(0.4 + d / 100.0 + rng.normal(0, 0.05, n_per), 0, 1)
        rows[-1]["risk_label"] = ["NORMAL", "WARNING", "CRITICAL"][cls]
    df = pd.concat(rows, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["split"] = np.where(np.arange(len(df)) % 10 < 6, "train",
                           np.where(np.arange(len(df)) % 10 < 8, "validation", "test"))
    return df


def test_baselines_use_the_same_feature_matrix_as_xgboost() -> None:
    df = _frame()
    xgb = train_risk_model(df)
    baselines = fit_baselines(df[df.split == "train"], xgb.features)
    assert baselines.features == xgb.features, "/: baselines share XGBoost's feature matrix"


def test_logistic_and_rf_predict_on_test() -> None:
    df = _frame()
    xgb = train_risk_model(df)
    b = fit_baselines(df[df.split == "train"], xgb.features)
    test = df[df.split == "test"]
    pl = b.predict_logistic(test)
    pr = b.predict_random_forest(test)
    assert set(pl) <= set(b.classes) and set(pr) <= set(b.classes)
    acc_rf = float((pr == test.risk_label.to_numpy()).mean())
    assert acc_rf > 0.8


def test_threshold_rule_persistence_requires_consecutive_windows() -> None:
    df = pd.DataFrame(
        {
            "event_id": "E_0_000",
            "node_id": "V0000",
            "displacement": [30.0] * 6,
            "velocity": [0.3] * 6,
            "acceleration": [0.001] * 6,
        }
    )
    # velocity above warning threshold for all 6 windows: after 3-window
    # persistence the label must be WARNING (never CRITICAL — acceleration low)
    pred = threshold_rule_predict(df, velocity_warning=0.05, persistence_windows=3)
    assert list(pred[:2]) == ["NORMAL", "NORMAL"], "first windows cannot carry the label yet"
    assert set(pred[2:]) == {"WARNING"}
    # a single hot window inside a calm series must NOT flag (calm = low level
    # AND low velocity; the level alone above score_warning counts as hot)
    df2 = pd.DataFrame(
        {
            "event_id": ["E", "E", "E", "E"],
            "node_id": ["V"] * 4,
            "displacement": [0.0, 30.0, 0.0, 0.0],
            "velocity": [0.001, 0.3, 0.001, 0.001],
            "acceleration": [0.0001] * 4,
        }
    )
    pred2 = threshold_rule_predict(df2, velocity_warning=0.05, persistence_windows=3)
    assert set(pred2) == {"NORMAL"}, "an isolated spike must not sustain a label"


def test_threshold_rule_critical_needs_acceleration() -> None:
    df = pd.DataFrame(
        {
            "event_id": "E_0_000",
            "node_id": "V0000",
            "displacement": [30.0] * 6,
            "velocity": [0.3] * 6,
            "acceleration": [0.05] * 6,
        }
    )
    pred = threshold_rule_predict(df, velocity_warning=0.05, accel_critical=0.02, persistence_windows=3)
    assert set(pred[2:]) == {"CRITICAL"}


def test_calibration_chooses_validation_best_params() -> None:
    df = _frame(seed=1)
    val = df[df.split == "validation"]
    best = calibrate_threshold_rule(val)
    assert "params" in best and best["macro_f1"] >= 0.0
    # calibrated params must reproduce the validation score exactly
    pred = threshold_rule_predict(val, **best["params"])
    from sklearn.metrics import f1_score

    assert f1_score(val["risk_label"], pred, average="macro") == pytest.approx(best["macro_f1"])


def test_baselines_missing_features_raise() -> None:
    df = _frame()
    with pytest.raises(BaselinesError, match="missing"):
        fit_baselines(df[df.split == "train"], features=["does_not_exist"])


def test_threshold_rule_missing_columns_raise() -> None:
    df = _frame().drop(columns=["acceleration"])
    with pytest.raises(BaselinesError):
        threshold_rule_predict(df)
