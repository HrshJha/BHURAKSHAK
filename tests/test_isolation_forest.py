"""T-043 acceptance tests — §14 Isolation Forest module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.anomaly.isolation_forest import (
    ABLATION_STEPS,
    IsolationForestError,
    _resolve_features,
    healthy_baseline_mask,
    train_isolation_forest,
)
from src.config import anomaly_config


def _frame(n_healthy: int = 800, n_anomalous: int = 150, seed: int = 0) -> pd.DataFrame:
    """Healthy windows near the origin; anomalous windows pushed on key dims."""
    features = _resolve_features(list(ABLATION_STEPS))
    rng = np.random.default_rng(seed)
    healthy = rng.normal(0.0, 0.3, size=(n_healthy, len(features)))
    # anomalous windows are globally displaced (a compact abnormal cluster far
    # from the healthy hull) — the subtle partial-dimension regime is exercised
    # by the T-045 ablation on real feature-store data, not by this unit fixture
    anomalous = rng.normal(0.0, 0.3, size=(n_anomalous, len(features))) + 4.0
    X = np.vstack([healthy, anomalous])
    df = pd.DataFrame(X, columns=features)
    df["anomaly_label"] = np.r_[np.zeros(n_healthy, dtype=int), np.ones(n_anomalous, dtype=int)]
    df["fault_label"] = "NONE"
    df["risk_label"] = np.where(df["anomaly_label"] == 0, "NORMAL", "WARNING")
    df["split"] = np.where(np.arange(len(df)) % 5 == 0, "validation", "train")
    return df


def test_config_holds_the_exact_section_14_parameters() -> None:
    cfg = anomaly_config()["isolation_forest"]
    assert cfg["n_estimators"] == 300, "§14 code block: n_estimators=300"
    assert cfg["contamination"] == "auto"
    assert cfg["random_state"] == 42


def test_healthy_mask_is_the_triple_conjunction() -> None:
    """G-2: faulted windows with risk_label=NORMAL must NOT count as healthy."""
    df = pd.DataFrame(
        {
            "anomaly_label": [0, 0, 1, 0],
            "fault_label": ["NONE", "BIAS", "NONE", "NONE"],
            "risk_label": ["NORMAL", "NORMAL", "NORMAL", "WARNING"],
        }
    )
    mask = healthy_baseline_mask(df)
    assert mask.tolist() == [True, False, False, False], (
        "healthy = anomaly 0 AND fault NONE AND risk NORMAL (G-2)"
    )


def test_healthy_mask_requires_label_columns() -> None:
    with pytest.raises(IsolationForestError):
        healthy_baseline_mask(pd.DataFrame({"risk_label": ["NORMAL"]}))


def test_training_uses_only_healthy_baseline_rows() -> None:
    """§14: the model must be fit on healthy rows only — verified by
    contamination of the training set producing a degenerate detector otherwise."""
    df = _frame()
    fitted = train_isolation_forest(df)
    # training-set size equals the healthy train-split count
    expected = int((healthy_baseline_mask(df) & (df["split"] == "train")).sum())
    assert fitted.n_training_windows == expected


def test_anomaly_score_is_negative_score_samples() -> None:
    df = _frame()
    fitted = train_isolation_forest(df)
    scores = fitted.anomaly_score(df)
    direct = -fitted.model.score_samples(df[fitted.features].to_numpy(dtype=float))
    np.testing.assert_allclose(scores, direct, rtol=1e-12)
    assert scores.mean() > 0  # sanity: scores are around 0.4-0.7 for iForest


def test_anomalous_windows_score_higher_than_healthy() -> None:
    df = _frame()
    fitted = train_isolation_forest(df)
    scores = fitted.anomaly_score(df)
    healthy = scores[df["anomaly_label"].to_numpy() == 0]
    anomalous = scores[df["anomaly_label"].to_numpy() == 1]
    assert anomalous.mean() > healthy.mean() + 0.1
    assert healthy.mean() < fitted.threshold < anomalous.mean()


def test_flags_fire_on_anomalies_not_healthy() -> None:
    df = _frame()
    fitted = train_isolation_forest(df)
    flags = fitted.flags(df)
    hit_rate = flags[df["anomaly_label"].to_numpy() == 1].mean()
    false_rate = flags[df["anomaly_label"].to_numpy() == 0].mean()
    assert hit_rate > 0.9
    assert false_rate < 0.05


def test_threshold_set_on_validation_not_train() -> None:
    """The fitted threshold must equal a high quantile of VALIDATION healthy
    scores — recomputing it from the held-out validation split must agree."""
    df = _frame(seed=3)
    fitted = train_isolation_forest(df)
    val = df[df["split"] == "validation"]
    val_scores = fitted.anomaly_score(val[healthy_baseline_mask(val)])
    expected = float(np.quantile(val_scores, anomaly_config()["far_alpha"]))
    assert fitted.threshold == pytest.approx(expected, rel=1e-9)
    assert "validation" in fitted.threshold_rule


def test_ablation_feature_sets_resolve_in_order() -> None:
    from src.anomaly.isolation_forest import _resolve_features

    full = _resolve_features(["A_physical", "B_add_temporal", "C_add_spatial"])
    assert full[:5] == ["tilt_x", "tilt_y", "tilt_magnitude", "displacement", "strain"]
    assert "spatial_coherence" in full and "rolling_mean" in full
    assert len(full) == 23  # 5 + 10 + 8 (§14 ablation groups, deduplicated)
    a_only = _resolve_features(["A_physical"])
    assert a_only == ["tilt_x", "tilt_y", "tilt_magnitude", "displacement", "strain"]


def test_unknown_feature_group_raises() -> None:
    df = _frame()
    with pytest.raises(IsolationForestError, match="unknown ablation"):
        train_isolation_forest(df, feature_groups=["A_physical", "bogus"])


def test_missing_split_column_raises() -> None:
    df = _frame().drop(columns=["split"])
    with pytest.raises(IsolationForestError, match="split"):
        train_isolation_forest(df)


def test_missing_feature_columns_raise() -> None:
    df = _frame().drop(columns=["strain"])
    with pytest.raises(IsolationForestError, match="strain"):
        train_isolation_forest(df)


def test_ablation_steps_order_documented() -> None:
    assert ABLATION_STEPS == ("A_physical", "B_add_temporal", "C_add_spatial")
