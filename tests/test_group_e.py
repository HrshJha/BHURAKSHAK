"""T-039 acceptance tests — Feature Group E (sensor health)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.group_e_health import GROUP_E_FEATURES, emit_group_e
from src.features.windowing import build_windows

INTERVAL = 10.0 / 60.0


def _frames(stuck_at: int | None = None, drift: bool = False, drop_steps: tuple[int, ...] = ()):
    n = 144
    ts = INTERVAL * np.arange(n, dtype=float)
    rng = np.random.default_rng(3)
    disp = 0.02 * rng.standard_normal(n)
    if drift:
        disp = disp + np.linspace(0.0, 1.5, n)
    if stuck_at is not None:
        disp[stuck_at : stuck_at + 10] = disp[stuck_at]
    full = pd.DataFrame(
        {
            "event_id": "E1",
            "node_id": "V0001",
            "timestamp": ts,
            "displacement": disp,
            "battery": np.linspace(4.1, 3.6, n),
            "RSSI": -95.0 + 2.0 * rng.standard_normal(n),
            "SNR": rng.standard_normal(n),
            "packet_loss": (rng.random(n) > 0.98).astype(float),
        }
    )
    # windows are defined on the COMPLETE grid; the raw frame then carries gaps
    windowed = build_windows(full).df
    keep = np.ones(n, dtype=bool)
    keep[list(drop_steps)] = False
    return full[keep].reset_index(drop=True), windowed


def test_feature_names_match_section_13() -> None:
    assert GROUP_E_FEATURES == (
        "battery",
        "RSSI",
        "SNR",
        "packet_loss",
        "missing_ratio",
        "stuck_sensor_flag",
        "drift_score",
    )


def test_telemetry_means_flow_through() -> None:
    raw, windowed = _frames()
    out = emit_group_e(windowed, raw)
    row = out.iloc[0]
    assert row["battery"] == pytest.approx(windowed["battery_mean"].iloc[0])
    assert row["RSSI"] == pytest.approx(windowed["RSSI_mean"].iloc[0])
    assert row["SNR"] == pytest.approx(windowed["SNR_mean"].iloc[0])
    assert row["packet_loss"] == pytest.approx(windowed["packet_loss_mean"].iloc[0])


def test_stuck_flag_fires_for_stuck_window() -> None:
    raw, windowed = _frames(stuck_at=65)  # steps 65..74 inside window 6 (60..69)/7 (70..79)
    out = emit_group_e(windowed, raw)
    assert out["stuck_sensor_flag"].sum() >= 1, "a 10-step stuck run must flag at least one window"
    # a clean series has no stuck windows
    raw2, windowed2 = _frames()
    out2 = emit_group_e(windowed2, raw2)
    assert out2["stuck_sensor_flag"].sum() == 0


def test_drift_score_high_for_drift_series_low_for_clean() -> None:
    raw, windowed = _frames(drift=True)
    out = emit_group_e(windowed, raw)
    raw2, windowed2 = _frames()
    out2 = emit_group_e(windowed2, raw2)
    assert out["drift_score"].mean() > out2["drift_score"].mean() + 1.0
    assert (out["drift_score"] > 3.5).any(), "the §10 DRIFT fault must be visible in windows"


def test_missing_ratio_counts_absent_steps() -> None:
    raw, windowed = _frames(drop_steps=(5, 6))  # two missing steps inside window 0
    out = emit_group_e(windowed, raw)
    row0 = out.iloc[0]
    assert row0["missing_ratio"] == pytest.approx(2 / 60)


def test_missing_ratio_zero_when_complete() -> None:
    raw, windowed = _frames()
    out = emit_group_e(windowed, raw)
    assert (out["missing_ratio"] < 1e-9).all()


def test_behaviour_features_nan_when_series_absent() -> None:
    raw, windowed = _frames()
    other = windowed.assign(node_id="V9999")
    out = emit_group_e(pd.concat([windowed, other], ignore_index=True), raw)
    tail = out[out["node_id"] == "V9999"]
    assert tail["missing_ratio"].isna().all()
    assert tail["stuck_sensor_flag"].isna().all()


def test_without_raw_frame_behaviour_features_default_to_zero() -> None:
    _raw, windowed = _frames()
    out = emit_group_e(windowed)
    assert (out["missing_ratio"] == 0.0).all()
    assert (out["stuck_sensor_flag"] == 0.0).all()
    assert (out["drift_score"] == 0.0).all()


def test_missing_telemetry_sources_raise() -> None:
    _raw, windowed = _frames()
    with pytest.raises(ValueError, match="battery_mean"):
        emit_group_e(windowed.drop(columns=["battery_mean"]))


def test_one_row_per_window() -> None:
    raw, windowed = _frames()
    out = emit_group_e(windowed, raw)
    assert len(out) == len(windowed) == 9
