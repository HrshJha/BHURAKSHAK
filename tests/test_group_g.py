""" acceptance tests — Feature Group G (DGPS/GNSS), Group G names."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.group_g_dgps import GROUP_G_FEATURES, GroupGError, emit_group_g

EPOCH = pd.Timestamp("2026-03-02")
H = {"Mar09": 168.0, "Mar21": 456.0, "Apr02": 744.0}


def _nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {"node_id": ["V_A", "V_B", "V_C"], "x": [0.0, 100.0, 400.0], "y": [0.0, 0.0, 0.0]}
    )


def _survey() -> pd.DataFrame:
    rows = []
    for pid, x, y, vals in (
        ("CP0", 5.0, 0.0, [0.0, -5.0, -9.0]),
        ("CP1", 108.0, 0.0, [0.0, 1.0, 2.0]),
    ):
        for d, v in zip(["2026-03-09", "2026-03-21", "2026-04-02"], vals, strict=True):
            rows.append(
                {
                    "point_id": pid,
                    "x": x,
                    "y": y,
                    "observation_timestamp": d,
                    "vertical_displacement_mm": v,
                    "horizontal_displacement_mm": 0.3,
                }
            )
    return pd.DataFrame(rows)


def _windowed(nodes: list[str], keys: list[str]) -> pd.DataFrame:
    rows = [
        {
            "event_id": "E1",
            "node_id": n,
            "window_index": i,
            "window_timestamp": H[k],
            "displacement_mean": 0.0,
        }
        for i, k in enumerate(keys)
        for n in nodes
    ]
    return pd.DataFrame(rows)


def test_feature_names_match_section_13() -> None:
    assert GROUP_G_FEATURES == (
        "vertical_displacement",
        "horizontal_displacement",
        "velocity",
        "acceleration",
        "mesh_vs_dgps_residual",
    )


def test_control_point_surfaces_at_its_node_window() -> None:
    win = _windowed(["V_A", "V_B"], ["Mar21"])
    out = emit_group_g(_survey(), win, _nodes(), epoch=EPOCH)
    a = out[out.node_id == "V_A"].iloc[0]
    assert a["vertical_displacement"] == pytest.approx(-5.0)
    assert a["horizontal_displacement"] == pytest.approx(0.3)
    assert np.isfinite(a["velocity"]) and a["velocity"] < 0, "CP0 subsides → negative velocity"


def test_nodes_without_control_points_stay_nan() -> None:
    win = _windowed(["V_A", "V_B", "V_C"], ["Mar21"])
    out = emit_group_g(_survey(), win, _nodes(), epoch=EPOCH)
    c = out[out.node_id == "V_C"].iloc[0]
    assert np.isnan(c["vertical_displacement"])
    assert np.isnan(c["mesh_vs_dgps_residual"])


def test_windows_outside_dgps_tolerance_yield_nan() -> None:
    """A window hours away from any survey pass cannot see the survey."""
    win = _windowed(["V_A"], ["Mar21"])
    win["window_timestamp"] = 24.0 * 200  # far from all three survey dates
    out = emit_group_g(_survey(), win, _nodes(), epoch=EPOCH)
    row = out.iloc[0]
    assert np.isnan(row["vertical_displacement"])
    assert np.isnan(row["velocity"])


def test_velocity_and_acceleration_are_point_trend_terms() -> None:
    """CP0 falls 0→−5→−9 mm at days 0/12/24: least-squares slope =
 −0.375 mm/day → ≈ −137 mm/yr (the LS fit, not the chord average)."""
    win = _windowed(["V_A"], ["Mar09", "Mar21", "Apr02"])
    out = emit_group_g(_survey(), win, _nodes(), epoch=EPOCH).sort_values("window_index")
    vel = out["velocity"].to_numpy(dtype=float)
    assert np.allclose(vel, vel[0])  # constant fit over the series
    assert vel[0] == pytest.approx(-0.375 * 365.25, rel=0.01)


def test_datetime_microsecond_resolution_keeps_velocity_units() -> None:
    survey = _survey()
    survey["observation_timestamp"] = pd.to_datetime(
        survey["observation_timestamp"]
    ).astype("datetime64[us]")
    win = _windowed(["V_A"], ["Mar09", "Mar21", "Apr02"])
    out = emit_group_g(survey, win, _nodes(), epoch=EPOCH).sort_values("window_index")
    assert out["velocity"].iloc[0] == pytest.approx(-0.375 * 365.25, rel=0.01)


def test_residual_rides_the_window_mesh_estimate() -> None:
    win = _windowed(["V_A", "V_B"], ["Mar21"])
    win.loc[win.node_id == "V_A", "displacement_mean"] = -4.6  # mesh estimate
    win.loc[win.node_id == "V_B", "displacement_mean"] = 1.4
    out = emit_group_g(_survey(), win, _nodes(), epoch=EPOCH)
    a = out[out.node_id == "V_A"].iloc[0]
    assert a["mesh_vs_dgps_residual"] == pytest.approx(-4.6 - (-5.0), abs=1e-9)
    b = out[out.node_id == "V_B"].iloc[0]
    assert b["mesh_vs_dgps_residual"] == pytest.approx(1.4 - 1.0, abs=1e-9)


def test_rows_align_with_window_frame() -> None:
    win = _windowed(["V_B", "V_A", "V_C"], ["Mar09", "Mar21"])
    out = emit_group_g(_survey(), win, _nodes(), epoch=EPOCH)
    assert len(out) == len(win)
    assert list(out.columns[:3]) == ["event_id", "node_id", "window_index"]
    assert out.groupby("window_index").size().nunique() == 1


def test_missing_window_keys_raise() -> None:
    with pytest.raises(GroupGError, match="window_timestamp"):
        emit_group_g(_survey(), _windowed(["V_A"], ["Mar21"]).drop(columns=["window_timestamp"]), _nodes(), epoch=EPOCH)


def test_missing_mesh_estimate_raises() -> None:
    with pytest.raises(GroupGError, match="displacement_mean"):
        emit_group_g(_survey(), _windowed(["V_A"], ["Mar21"]).drop(columns=["displacement_mean"]), _nodes(), epoch=EPOCH)


def test_velocity_needs_three_observations() -> None:
    two = _survey().iloc[:2]  # CP0 with only two passes
    win = _windowed(["V_A"], ["Mar21"])
    out = emit_group_g(two, win, _nodes(), epoch=EPOCH)
    assert np.isnan(out.iloc[0]["velocity"])
    assert np.isfinite(out.iloc[0]["vertical_displacement"])
