"""T-046 acceptance tests — §15 XGBoost risk classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.xgboost_model import (
    EXTRA_SIGNALS,
    MODEL_INPUT_GROUPS,
    XGBoostModelError,
    train_risk_model,
)

FEATURES = ["displacement", "velocity", "acceleration", "tilt_x", "physics_residual", "anomaly_score"]


def _frame(n: int = 2400, seed: int = 0) -> pd.DataFrame:
    """3 separable risk classes driven by displacement/velocity physics."""
    rng = np.random.default_rng(seed)
    n_per = n // 3
    rows = []
    for cls, (mu_d, mu_v) in enumerate(((2.0, 0.01), (18.0, 0.08), (42.0, 0.25))):
        d = rng.normal(mu_d, 2.0, n_per)
        v = rng.normal(mu_v, 0.01, n_per)
        rows.append(
            pd.DataFrame(
                {
                    "displacement": d,
                    "velocity": v,
                    "acceleration": 0.01 * v + rng.normal(0, 0.002, n_per),
                    "tilt_x": d / 400.0,
                    "physics_residual": rng.normal(0, 1.0, n_per),
                    "anomaly_score": np.clip(0.4 + d / 100.0 + rng.normal(0, 0.05, n_per), 0, 1),
                    "risk_label": ["NORMAL", "WARNING", "CRITICAL"][cls],
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["split"] = np.where(np.arange(len(df)) % 10 < 6, "train",
                           np.where(np.arange(len(df)) % 10 < 8, "validation", "test"))
    return df


def test_model_input_contract_lists_groups_and_signals() -> None:
    # T-070: the Phase-4 modalities (G DGPS / H Sentinel-1) are switchable
    # model inputs for the §25 ablation; T-077 adds the §16 forecast features
    # (I_forecast) — frames without their columns are unaffected (resolver
    # skips absent groups).
    assert MODEL_INPUT_GROUPS[:6] == ("A_physical", "B_temporal", "C_spatial", "D_vibration", "E_sensor_health", "F_physics")
    assert MODEL_INPUT_GROUPS[6:] == ("G_dgps", "H_insar", "I_forecast")
    assert EXTRA_SIGNALS == ("anomaly_score", "physics_residual"), "§15: IF score + physics residual feed XGBoost"


def test_training_reaches_three_classes_and_predicts_proba() -> None:
    df = _frame()
    fitted = train_risk_model(df)
    assert fitted.classes == ["CRITICAL", "NORMAL", "WARNING"]
    proba = fitted.predict_proba(df)
    assert proba.shape == (len(df), 3)
    # XGBoost computes in float32: sums equal 1 to ~1e-6, not 1e-12
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5), "per-class probabilities must sum to 1"


def test_predictions_respect_split_discipline() -> None:
    df = _frame()
    fitted = train_risk_model(df)
    test = df[df["split"] == "test"]
    pred = fitted.predict(test)
    acc = float((pred == test["risk_label"].to_numpy()).mean())
    assert acc > 0.9, f"held-out accuracy too low for a separable fixture: {acc}"


def test_anomaly_score_is_a_model_input() -> None:
    df = _frame()
    fitted = train_risk_model(df)
    assert "anomaly_score" in fitted.features, "§15: anomaly_score must be among the inputs"
    assert "physics_residual" in fitted.features


def test_missing_split_or_label_raises() -> None:
    df = _frame()
    with pytest.raises(XGBoostModelError):
        train_risk_model(df.drop(columns=["split"]))
    with pytest.raises(XGBoostModelError):
        train_risk_model(df.drop(columns=["risk_label"]))


def test_non_three_class_targets_rejected() -> None:
    df = _frame()
    df["risk_label"] = df["risk_label"].replace({"CRITICAL": "WARNING"})
    with pytest.raises(XGBoostModelError, match="3 classes"):
        train_risk_model(df)


def test_missing_features_at_inference_raise() -> None:
    df = _frame()
    fitted = train_risk_model(df)
    with pytest.raises(XGBoostModelError, match="missing"):
        fitted.predict_proba(df.drop(columns=["anomaly_score"]))


def test_config_driven_hyperparameters() -> None:
    from src.config import risk_model_config

    cfg = risk_model_config()["xgboost"]
    assert cfg["random_state"] == 42
    assert cfg["n_estimators"] > 0
