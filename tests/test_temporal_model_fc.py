"""T-076 — §16 temporal forecasting model tests.

The acceptance spine: the model predicts a PHYSICAL quantity (future
displacement / tilt / deformation velocity) and NEVER "future danger"
directly; TCN/GRU/LSTM are selectable; LSTM is retained as the benchmark.
Torch-dependent paths are skipped cleanly if torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.temporal_model import (
    FORECASTABLE_CHANNELS,
    ForecastError,
    build_sequences,
    train_temporal_forecaster,
)

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
                base = {"EV_A": 2.0, "EV_B": 9.0}[event] * (w + 1) / 9.0
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


# --- the §16 acceptance spine ------------------------------------------------

def test_forecastable_channels_are_physical_only() -> None:
    assert FORECASTABLE_CHANNELS == ("displacement", "tilt_x", "tilt_y")
    for banned in ("risk", "alert", "probability", "label"):
        assert all(banned not in ch for ch in FORECASTABLE_CHANNELS)


def test_forecast_output_is_physical_table_not_risk(windowed: pd.DataFrame) -> None:
    fc = train_temporal_forecaster(windowed, horizons=(1,), epochs=2, architecture="lstm")
    out = fc.predict(windowed)
    assert {"event_id", "node_id", "channel", "horizon_steps", "predicted_value"} == set(out.rows.columns)
    assert set(out.rows["channel"].unique()) <= set(FORECASTABLE_CHANNELS)
    assert out.channel_units["displacement"] == "mm"


def test_forecast_cannot_carry_risk_columns() -> None:
    from src.forecasting.temporal_model import Forecast

    with pytest.raises(ForecastError, match="risk"):
        Forecast(rows=pd.DataFrame({"risk": [0.1], "predicted_value": [1.0]}))


def test_rejects_risk_like_channels(windowed: pd.DataFrame) -> None:
    bad = windowed.assign(risk_label_mean=1.0)
    with pytest.raises(ForecastError, match="missing"):
        train_temporal_forecaster(bad, channels=("risk_label",), epochs=1)


# --- architectures ------------------------------------------------------------

@pytest.mark.parametrize("architecture", ["lstm", "gru", "tcn"])
def test_architectures_selectable_and_train(windowed: pd.DataFrame, architecture: str) -> None:
    fc = train_temporal_forecaster(
        windowed, architecture=architecture, epochs=2, history_steps=4, horizons=(1, 2)
    )
    assert fc.architecture == architecture
    out = fc.predict(windowed)
    assert set(out.rows["horizon_steps"].unique()) == {1, 2}
    assert np.isfinite(out.rows["predicted_value"]).all()


def test_unknown_architecture_raises(windowed: pd.DataFrame) -> None:
    with pytest.raises(ForecastError, match="architecture"):
        train_temporal_forecaster(windowed, architecture="transformer", epochs=1)


def test_lstm_default_is_the_benchmark(windowed: pd.DataFrame) -> None:
    fc = train_temporal_forecaster(windowed, epochs=1)
    assert fc.architecture == "lstm"


# --- sequence construction ----------------------------------------------------

def test_build_sequences_shapes_and_targets(windowed: pd.DataFrame) -> None:
    x, y, keys = build_sequences(windowed, ("displacement",), history_steps=4, horizon_steps=2)
    # per series of 9 windows: 9 - 4 - 2 + 1 = 4 sequences; 4 series
    assert x.shape == (16, 4, 1)
    assert y.shape == (16, 1)
    assert len(keys) == 16
    # target of the first sequence of a series = value at index 4 + 2 - 1 = 5
    first = windowed[(windowed.event_id == "EV_A") & (windowed.node_id == "V0000")].sort_values("window_index")
    assert y[0, 0] == pytest.approx(float(first["displacement_mean"].iloc[5]), abs=1e-6)


def test_build_sequences_skip_gappy_series(windowed: pd.DataFrame) -> None:
    gappy = windowed.copy()
    gappy.loc[gappy.index[:4], "displacement_mean"] = np.nan  # first series head is gone
    x, _, keys = build_sequences(gappy, ("displacement",), history_steps=4, horizon_steps=2)
    assert len(keys) < 16  # some sequences dropped, none invented


def test_build_sequences_rejects_missing_channel(windowed: pd.DataFrame) -> None:
    with pytest.raises(ForecastError, match="missing"):
        build_sequences(windowed, ("strain",), history_steps=4, horizon_steps=1)


# --- training discipline ------------------------------------------------------

def test_train_requires_train_sequences(windowed: pd.DataFrame) -> None:
    empty = windowed[windowed.index < 0]
    with pytest.raises(ForecastError, match="no trainable"):
        train_temporal_forecaster(empty, epochs=1)


def test_short_series_produce_no_forecast(windowed: pd.DataFrame) -> None:
    fc = train_temporal_forecaster(windowed, epochs=2, history_steps=4)
    tiny = windowed[windowed.window_index < 2]  # 2 windows < history 4
    out = fc.predict(tiny)
    assert out.rows.empty


def test_forecast_reproduces_trend_direction(windowed: pd.DataFrame) -> None:
    """A trained forecaster should predict a LARGER future displacement for
    the rising series than for the flat one (sanity, not accuracy)."""
    fc = train_temporal_forecaster(windowed, architecture="gru", epochs=15, history_steps=4, horizons=(1,))
    out = fc.predict(windowed)
    disp = out.rows[(out.rows.channel == "displacement") & (out.rows.horizon_steps == 1)]
    rising = disp[(disp.event_id == "EV_B") & (disp.node_id == "V0000")]["predicted_value"].iloc[0]
    flat = disp[(disp.event_id == "EV_A") & (disp.node_id == "V0009")]["predicted_value"].iloc[0]
    # EV_B/V0000 climbs to ~9 mm; EV_A/V0009 stays near 0 — direction must hold
    assert rising > flat + 1.0
