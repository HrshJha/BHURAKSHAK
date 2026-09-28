"""T-077 — §16 forecast→risk-layer bridge tests.

The acceptance spine: forecasted PHYSICAL values enter the XGBoost risk layer
as FEATURES, and the forecaster never emits a risk state directly. Torch-
dependent paths are skipped cleanly if torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.forecast_to_risk import (
    FORECAST_FEATURE_PREFIX,
    LeakageError,
    assert_no_forecast_leakage,
    forecast_all_origins,
    forecast_columns_of,
    forecast_feature_names,
    forecasts_to_feature_frame,
    is_forecast_feature,
    join_forecast_features,
)
from src.forecasting.temporal_model import ForecastError, train_temporal_forecaster

torch = pytest.importorskip("torch", reason="torch not installed")


@pytest.fixture()
def windowed() -> pd.DataFrame:
    """Three series with a linear + sinusoidal displacement signal."""
    rng = np.random.default_rng(7)
    rows = []
    for event in ("EV_A", "EV_B"):
        for node in ("V0000", "V0009"):
            for w in range(9):
                t = w * 0.6
                base = {"EV_A": 2.0, "EV_B": 12.0}[event] * (w + 1) / 9.0
                rows.append(
                    {
                        "event_id": event,
                        "node_id": node,
                        "window_index": w,
                        "window_timestamp": t,
                        "displacement_mean": base + 0.4 * np.sin(2 * np.pi * t / 3.6) + rng.normal(0, 0.05),
                        "tilt_x_mean": 0.05 * base + rng.normal(0, 0.005),
                        "tilt_y_mean": 0.03 * base + rng.normal(0, 0.005),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def feature_frame(windowed: pd.DataFrame) -> pd.DataFrame:
    """A §15-shaped model frame: window keys + two risk features + labels."""
    rows = []
    for (event, node), g in windowed.groupby(["event_id", "node_id"], sort=False):
        for _, r in g.iterrows():
            d = r["displacement_mean"]
            rows.append(
                {
                    "event_id": event,
                    "node_id": node,
                    "window_index": int(r["window_index"]),
                    "window_timestamp": r["window_timestamp"],
                    "displacement": d,
                    "velocity": d / 10.0,
                    "risk_label": "CRITICAL" if d > 6 else ("WARNING" if d > 3 else "NORMAL"),
                    "split": "train" if r["window_index"] < 6 else ("validation" if r["window_index"] < 8 else "test"),
                }
            )
    return pd.DataFrame(rows)


# --- naming contract ----------------------------------------------------------

def test_forecast_feature_names_and_recogniser() -> None:
    names = forecast_feature_names(("displacement", "tilt_x"), (1, 6))
    assert names == ["forecast_displacement_h1", "forecast_displacement_h6",
                     "forecast_tilt_x_h1", "forecast_tilt_x_h6"]
    assert all(is_forecast_feature(n) for n in names)
    assert is_forecast_feature(f"{FORECAST_FEATURE_PREFIX}tilt_y_h24")
    assert not is_forecast_feature("displacement")
    assert not is_forecast_feature("forecast_displacement")  # no horizon suffix
    assert not is_forecast_feature("forecast_displacement_horizon")  # suffix must be _h<digits>


def test_forecast_columns_of_is_deterministic_and_ordered(feature_frame: pd.DataFrame) -> None:
    joined = feature_frame.assign(forecast_tilt_x_h1=0.1, forecast_displacement_h24=1.0)
    cols = forecast_columns_of(joined)
    # column order of the frame, not set iteration — deterministic for the model
    assert cols == ["forecast_tilt_x_h1", "forecast_displacement_h24"]


# --- every-origin forecasting --------------------------------------------------

def test_forecast_all_origins_covers_every_origin(windowed: pd.DataFrame) -> None:
    fc = train_temporal_forecaster(windowed, epochs=2, history_steps=4, horizons=(1, 2))
    rows = forecast_all_origins(fc, windowed)
    # origins = history-1 .. n-1 with per-horizon target filtering: h1 reaches
    # origin 7 (target 8 = last), h2 stops at origin 6 → union 3..7
    assert sorted(rows["window_index"].unique().tolist()) == [3, 4, 5, 6, 7]
    # horizon 2 rows stop at origin 6 (target 8 = last); horizon 1 rows reach origin 7
    h1 = rows[rows.horizon_steps == 1]
    h2 = rows[rows.horizon_steps == 2]
    assert sorted(h1["window_index"].unique().tolist()) == [3, 4, 5, 6, 7]
    assert (h2["window_index"] <= 6).all()
    # target alignment: target = origin + horizon
    assert (rows["target_window_index"] == rows["window_index"] + rows["horizon_steps"]).all()
    # channels are the forecaster's, values physical
    assert set(rows["channel"].unique()) == set(fc.channels)
    assert rows["predicted_value"].notna().all()


def test_forecast_all_origins_never_uses_future_values(windowed: pd.DataFrame) -> None:
    """Causality: permuting FUTURE windows of a series must not change the
    forecast made from an earlier origin."""
    fc = train_temporal_forecaster(windowed, epochs=2, history_steps=4, horizons=(1, 2))
    rows_a = forecast_all_origins(fc, windowed)
    shuffled = windowed.copy()
    # reverse every series' windows AFTER origin 3 (the first origin's history end)
    parts = []
    for _, g in shuffled.groupby(["event_id", "node_id"], sort=False):
        g = g.sort_values("window_index")
        head, tail = g[g.window_index <= 3], g[g.window_index > 3]
        parts.append(pd.concat([head, tail.iloc[::-1]], ignore_index=True))
    rows_b = forecast_all_origins(fc, pd.concat(parts, ignore_index=True))
    key = ["event_id", "node_id", "window_index", "channel", "horizon_steps"]
    a = rows_a.set_index(key)["predicted_value"].sort_index()
    b = rows_b.set_index(key)["predicted_value"].sort_index()
    shared = a.index.intersection(b.index)
    pd.testing.assert_series_equal(a.loc[shared], b.loc[shared], check_names=False)


def test_forecast_all_origins_short_series_yield_nothing(windowed: pd.DataFrame) -> None:
    fc = train_temporal_forecaster(windowed, epochs=2, history_steps=4, horizons=(1,))
    tiny = windowed[windowed.window_index < 3]  # 3 windows < history 4
    assert forecast_all_origins(fc, tiny).empty


# --- pivoting + joining ---------------------------------------------------------

def test_pivot_produces_one_column_per_channel_horizon(windowed: pd.DataFrame) -> None:
    fc = train_temporal_forecaster(windowed, epochs=2, history_steps=4, horizons=(1, 2))
    rows = forecast_all_origins(fc, windowed)
    feats = forecasts_to_feature_frame(rows)
    assert list(feats.columns) == ["event_id", "node_id", "window_index",
                                   "forecast_displacement_h1", "forecast_displacement_h2",
                                   "forecast_tilt_x_h1", "forecast_tilt_x_h2",
                                   "forecast_tilt_y_h1", "forecast_tilt_y_h2"]
    assert feats.duplicated(["event_id", "node_id", "window_index"]).sum() == 0


def test_pivot_rejects_malformed_rows() -> None:
    with pytest.raises(ForecastError, match="missing columns"):
        forecasts_to_feature_frame(pd.DataFrame({"channel": ["displacement"]}))


def test_join_preserves_row_count_and_adds_forecast_columns(
    windowed: pd.DataFrame, feature_frame: pd.DataFrame
) -> None:
    fc = train_temporal_forecaster(windowed, epochs=2, history_steps=4, horizons=(1, 2))
    rows = forecast_all_origins(fc, windowed)
    joined = join_forecast_features(feature_frame, rows)
    assert len(joined) == len(feature_frame)  # left join, no duplication
    assert {"forecast_displacement_h1", "forecast_displacement_h2",
            "forecast_tilt_x_h1", "forecast_tilt_x_h2",
            "forecast_tilt_y_h1", "forecast_tilt_y_h2"} <= set(joined.columns)


def test_join_with_empty_forecasts_is_a_noop(feature_frame: pd.DataFrame) -> None:
    empty = pd.DataFrame(columns=["event_id", "node_id", "window_index"])
    joined = join_forecast_features(feature_frame, empty)
    assert len(joined) == len(feature_frame)
    assert forecast_columns_of(joined) == []


# --- leakage discipline -----------------------------------------------------------

def test_leakage_assert_passes_on_complete_frame(
    windowed: pd.DataFrame, feature_frame: pd.DataFrame
) -> None:
    fc = train_temporal_forecaster(windowed, epochs=2, history_steps=4, horizons=(1, 2))
    rows = forecast_all_origins(fc, windowed)
    joined = join_forecast_features(feature_frame, rows)
    # the feature frame stops at window 8; h1/h2 forecasts cover origins 3..7/3..6 —
    # every model row's forecast column must be observed except series tails
    tails = joined[joined.window_index > 6]
    assert tails[["forecast_displacement_h1", "forecast_displacement_h2"]].isna().any().any()
    complete = joined.dropna(subset=forecast_columns_of(joined))
    assert_no_forecast_leakage(complete)  # no raise


def test_leakage_assert_refuses_nan_and_empty() -> None:
    bad = pd.DataFrame({"forecast_displacement_h1": [1.0, np.nan]})
    with pytest.raises(LeakageError, match="NaN"):
        assert_no_forecast_leakage(bad)
    with pytest.raises(LeakageError, match="no forecast feature columns"):
        assert_no_forecast_leakage(pd.DataFrame({"displacement": [1.0]}))


# --- §16 acceptance spine: forecasts enter the risk layer as FEATURES --------------

def test_forecasts_feed_xgboost_as_features_and_never_emit_risk(
    windowed: pd.DataFrame, feature_frame: pd.DataFrame
) -> None:
    from src.risk.xgboost_model import MODEL_INPUT_GROUPS, train_risk_model

    fc = train_temporal_forecaster(windowed, epochs=2, history_steps=4, horizons=(1, 2))
    rows = forecast_all_origins(fc, windowed)
    joined = join_forecast_features(feature_frame, rows)
    complete = joined.dropna(subset=forecast_columns_of(joined)).reset_index(drop=True)

    fitted = train_risk_model(complete)
    assert "I_forecast" in MODEL_INPUT_GROUPS
    fc_cols = [c for c in fitted.features if is_forecast_feature(c)]
    assert fc_cols, "risk model must consume the forecast features"
    proba = fitted.predict_proba(complete)
    assert proba.shape == (len(complete), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)

    # the forecaster itself: no risk state, label or probability anywhere
    assert not set(fc.channels) & {"risk_label", "risk_state", "probability", "alert"}


def test_risk_model_without_forecast_columns_is_unchanged() -> None:
    """Frames without forecast columns resolve exactly as before T-077."""
    from src.risk.xgboost_model import train_risk_model

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "displacement": rng.normal(20, 5, 300),
            "velocity": rng.normal(0.1, 0.01, 300),
            "risk_label": rng.choice(["NORMAL", "WARNING", "CRITICAL"], 300),
            "split": ["train"] * 200 + ["validation"] * 100,
        }
    )
    fitted = train_risk_model(df)
    assert not [c for c in fitted.features if is_forecast_feature(c)]
    assert fitted.features == ["displacement", "velocity"]
