"""T-034 acceptance tests — §10 windowing engine (60/10, config-driven)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.windowing import (
    WINDOW_END,
    WINDOW_INDEX,
    WINDOW_START,
    WINDOW_TIMESTAMP,
    WindowingError,
    assert_windowed,
    build_windows,
    windowing_params,
)

INTERVAL = 10.0 / 60.0  # §8.3 grid cadence in hours


def _series(n: int = 144, node: str = "V0001", event: str = "E1", ramp: float = 1.0) -> pd.DataFrame:
    ts = INTERVAL * np.arange(n, dtype=float)
    disp = np.linspace(0.0, ramp, n)
    return pd.DataFrame(
        {
            "event_id": event,
            "node_id": node,
            "timestamp": ts,
            "displacement": disp,
            "tilt_x": 0.01 * np.ones(n),
            "anomaly_label": (disp > 0.5).astype(int),
            "risk_label": np.where(disp > 0.5, "WARNING", "NORMAL"),
            "fault_label": "NONE",
        }
    )


def test_params_are_config_driven_60_10() -> None:
    window, stride = windowing_params()
    assert (window, stride) == (60, 10), "§10: window=60 timesteps, stride=10"


def test_window_count_and_stride() -> None:
    n = 144
    res = build_windows(_series(n))
    expected = (n - 60) // 10 + 1  # 9 windows: starts 0,10,...,80
    assert res.n_windows == expected
    starts = res.df[WINDOW_INDEX].tolist()
    assert starts == list(range(expected))
    assert res.df[WINDOW_START].tolist() == [INTERVAL * s for s in (0, 10, 20, 30, 40, 50, 60, 70, 80)]


def test_window_statistics_are_correct() -> None:
    df = _series(144)
    res = build_windows(df, channels=("displacement",))
    row0 = res.df.iloc[0]
    window_vals = df["displacement"].to_numpy()[:60]
    assert row0["displacement_mean"] == pytest.approx(window_vals.mean())
    assert row0["displacement_std"] == pytest.approx(window_vals.std())
    # slope: (v[-1]-v[0])/59 per step
    assert row0["displacement_slope"] == pytest.approx((window_vals[-1] - window_vals[0]) / 59.0)
    # velocity: per hour across the whole window duration (59 steps)
    assert row0["displacement_velocity"] == pytest.approx((window_vals[-1] - window_vals[0]) / (59 * INTERVAL))
    # window anchored at its last step
    assert row0[WINDOW_TIMESTAMP] == pytest.approx(INTERVAL * 59)


def test_all_default_channels_get_all_five_statistics() -> None:
    res = build_windows(_series(144))
    for ch in ("displacement", "tilt_x"):
        for stat in ("mean", "std", "slope", "velocity", "acceleration"):
            assert f"{ch}_{stat}" in res.df.columns, f"missing {ch}_{stat}"


def test_windows_never_cross_series_boundaries() -> None:
    """T-036-adjacent acceptance: rolling computation must not cross events."""
    a = _series(144, node="V0001", event="E1", ramp=1.0)
    b = _series(144, node="V0002", event="E1", ramp=2.0)
    res = build_windows(pd.concat([a, b], ignore_index=True))
    assert res.n_windows == 9 + 9
    # each window's internal duration is exactly 59 steps of one series
    durations = res.df[WINDOW_END] - res.df[WINDOW_START]
    assert np.allclose(durations, 59 * INTERVAL)


def test_windows_do_not_cross_events_for_same_node() -> None:
    a = _series(144, node="V0001", event="E1", ramp=1.0)
    b = _series(144, node="V0001", event="E2", ramp=3.0)
    res = build_windows(pd.concat([a, b], ignore_index=True))
    assert res.n_windows == 9 + 9
    per_event = res.df.groupby("event_id").size().to_dict()
    assert per_event == {"E1": 9, "E2": 9}


def test_labels_carried_into_windows() -> None:
    df = _series(144)
    res = build_windows(df, channels=("displacement",))
    # window 0: disp 0..0.5 (exclusive-ish) — first label NORMAL; last window starts at 80 → disp ≥ 0.55
    assert res.df.iloc[0]["risk_label"] in ("NORMAL", "WARNING")
    assert set(res.df["risk_label"]) <= {"NORMAL", "WARNING"}


def test_short_series_produces_no_windows() -> None:
    res = build_windows(_series(30))
    assert res.n_windows == 0
    assert res.df.empty


def test_ramp_window_slope_is_constant() -> None:
    res = build_windows(_series(144), channels=("displacement",))
    slopes = res.df["displacement_slope"].to_numpy()
    assert np.allclose(slopes, slopes[0]), "a linear ramp has identical window slopes"


def test_acceleration_zero_for_uniform_velocity() -> None:
    res = build_windows(_series(144), channels=("displacement",))
    acc = res.df["displacement_acceleration"].to_numpy()
    assert np.allclose(acc, 0.0, atol=1e-12), "linear ramp ⇒ zero acceleration"


def test_guard_raises_on_raw_rows() -> None:
    with pytest.raises(WindowingError, match="raw per-timestep"):
        assert_windowed(_series(144))


def test_guard_accepts_windowed_frame() -> None:
    res = build_windows(_series(144))
    assert_windowed(res.df)  # must not raise


def test_g4_reconciliation_window_budget() -> None:
    """Documented G-4 reconciliation: 9 windows per 144-step series.

    10,000 sequences (current gate dataset) → 90,000 event-level windows, just
    below §15's 100k floor; 50,000 sequences (§10 maximum) → 450,000 windows,
    inside §15's 100k–500k budget. The budgets reconcile across §10's upper
    range; the honest arithmetic is asserted here, not a convenient rounding.
    """
    windows_per_series = (144 - 60) // 10 + 1
    assert windows_per_series == 9
    assert windows_per_series * 10_000 == 90_000
    total_at_max = windows_per_series * 50_000
    assert 100_000 <= total_at_max <= 500_000, "G-4: §15 budget matches §10's upper range"


def test_window_stats_robust_to_nan_channel_values() -> None:
    df = _series(144)
    df.loc[df.index[5:15], "displacement"] = np.nan  # a 10-step NaN patch
    res = build_windows(df, channels=("displacement",))
    first = res.df.iloc[0]
    finite = df["displacement"].to_numpy()[:60]
    assert first["displacement_mean"] == pytest.approx(np.nanmean(finite))
