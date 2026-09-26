"""T-036 acceptance tests — Feature Group B (temporal)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.group_b_temporal import (
    GROUP_B_FEATURES,
    assert_no_boundary_crossing,
    emit_group_b,
)
from src.features.windowing import build_windows


def _multi_series() -> pd.DataFrame:
    """Two series with distinct levels so boundary crossing would be visible."""
    frames = []
    for node, base, event in (("V0001", 0.0, "E1"), ("V0002", 100.0, "E1"), ("V0001", -50.0, "E2")):
        n = 144
        ts = (10.0 / 60.0) * np.arange(n)
        disp = base + np.linspace(0.0, 5.0, n)
        frames.append(
            pd.DataFrame(
                {
                    "event_id": event,
                    "node_id": node,
                    "timestamp": ts,
                    "displacement": disp,
                    "tilt_x": 0.01 * np.ones(n),
                    "tilt_y": -0.01 * np.ones(n),
                }
            )
        )
    return build_windows(pd.concat(frames, ignore_index=True)).df


def test_all_section_13_group_b_features_emitted() -> None:
    out = emit_group_b(_multi_series())
    for feat in GROUP_B_FEATURES:
        assert feat in out.columns, f"missing §13 Group B feature {feat}"


def test_per_channel_temporal_features_emitted() -> None:
    out = emit_group_b(_multi_series())
    for ch in ("displacement", "tilt_x", "tilt_y"):
        for feat in GROUP_B_FEATURES:
            assert f"{ch}_{feat}" in out.columns


def test_rolling_stats_use_series_internal_history_only() -> None:
    out = emit_group_b(_multi_series())
    # V0001/E1 sits at base 0, V0002/E1 at base 100 — if a rolling mean ever
    # crossed the series boundary it would land between the two plateaus.
    e1 = out[(out["event_id"] == "E1") & (out["node_id"] == "V0001")]
    assert (e1["displacement_rolling_mean"] < 10.0).all(), "rolling mean leaked across node boundary"
    e1b = out[(out["event_id"] == "E1") & (out["node_id"] == "V0002")]
    assert (e1b["displacement_rolling_mean"] > 90.0).all()


def test_first_window_rolling_mean_equals_its_own_level() -> None:
    out = emit_group_b(_multi_series())
    first = out[(out["event_id"] == "E1") & (out["node_id"] == "V0001")].sort_values("window_index")
    assert first["displacement_rolling_mean"].iloc[0] == pytest.approx(first["displacement_mean"].iloc[0])


def test_no_boundary_crossing_guard_passes_on_clean_output() -> None:
    out = emit_group_b(_multi_series())
    assert_no_boundary_crossing(out)  # must not raise


def test_boundary_crossing_guard_detects_tampering() -> None:
    out = emit_group_b(_multi_series())
    tampered = out.copy()
    idx = tampered[(tampered["node_id"] == "V0001") & (tampered["event_id"] == "E1")].index[3]
    tampered.loc[idx, "displacement_rolling_mean"] = 12345.0
    with pytest.raises(AssertionError, match="boundary crossing"):
        assert_no_boundary_crossing(tampered)


def test_trend_is_total_change_over_window() -> None:
    out = emit_group_b(_multi_series())
    row = out[(out["event_id"] == "E1") & (out["node_id"] == "V0001")].sort_values("window_index")
    # displacement ramps 0→5 mm over 144 steps: slope ≈ 5/143 per step; trend over 60 steps ≈ 59 × slope
    expected = (5.0 / 143.0) * 59.0
    assert row["displacement_trend"].iloc[0] == pytest.approx(expected, rel=0.01)


def test_persistence_and_change_point_bounded() -> None:
    out = emit_group_b(_multi_series())
    assert out["displacement_persistence"].between(-1.0, 1.0).all()
    assert (out["displacement_change_point_score"] >= 0).all()


def test_grouped_by_series_even_when_input_shuffled() -> None:
    windowed = _multi_series().sample(frac=1.0, random_state=0)
    out = emit_group_b(windowed)
    assert_no_boundary_crossing(out)  # grouping must not depend on row order


def test_missing_channel_raises() -> None:
    windowed = _multi_series().drop(columns=["tilt_x_mean"])
    with pytest.raises(ValueError, match="tilt_x_mean"):
        emit_group_b(windowed)
