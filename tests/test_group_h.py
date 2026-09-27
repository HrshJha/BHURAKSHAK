"""T-061 acceptance tests — Feature Group H (InSAR), §13 Group H names."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.group_h_insar import GROUP_H_FEATURES, InSARFeatureError, emit_group_h

DATES = ["2026-01-08", "2026-01-20", "2026-02-13"]
EPOCH = pd.Timestamp("2026-01-01")
H0, H1, H2 = 168.0, 456.0, 1032.0  # acquisition hours on the window axis


def _joined() -> pd.DataFrame:
    """Two nodes with observations; V_C gets the InSAR signal."""
    rows = []
    for node, vals in (("V_A", [0.0, -1.0, -3.0]), ("V_B", [0.0, 0.5, 1.0])):
        for d, h, v in zip(DATES, (H0, H1, H2), vals, strict=True):
            rows.append(
                {
                    "node_id": node,
                    "date": d,
                    "observation_timestamp": pd.Timestamp(d),
                    "staleness_hours": 0.0,
                    "ps_id": f"PS_{node}",
                    "ps_distance_m": 0.0,
                    "insar_los_displacement_mm": v,
                    "insar_los_velocity_mm_yr": -20.0 if node == "V_A" else 5.0,
                    "insar_los_acceleration_mm_yr2": np.nan,
                    "insar_coherence": 8.0 / 10.0,
                    "insar_n_observations": 21,
                }
            )
    return pd.DataFrame(rows)


def _windowed(timestamps: list[float], nodes: list[str] | None = None) -> pd.DataFrame:
    nodes = nodes or ["V_A", "V_B"]
    rows = [
        {"event_id": "E1", "node_id": n, "window_index": i, "window_timestamp": t}
        for i, t in enumerate(timestamps)
        for n in nodes
    ]
    return pd.DataFrame(rows)


def _coords() -> pd.DataFrame:
    return pd.DataFrame(
        {"node_id": ["V_A", "V_B", "V_C"], "x": [0.0, 50.0, 400.0], "y": [0.0, 0.0, 0.0]}
    )


def test_feature_names_match_section_13() -> None:
    assert GROUP_H_FEATURES == (
        "LOS_displacement",
        "LOS_velocity",
        "LOS_acceleration",
        "cumulative_displacement",
        "coherence",
        "spatial_gradient",
        "local_hotspot_density",
    )


def test_as_of_semantics_pick_latest_observation_at_or_before_window() -> None:
    out = emit_group_h(_joined(), _windowed([H0, H1, H2]), _coords())
    a_w1 = out[(out.node_id == "V_A") & (out.window_index == 1)].iloc[0]
    assert a_w1["LOS_displacement"] == pytest.approx(-1.0)
    assert a_w1["observation_timestamp"] == pd.Timestamp(DATES[1])
    # a window BETWEEN acquisitions holds the most recent observation
    a_mid = out[out.node_id == "V_A"].iloc[-1]
    out2 = emit_group_h(_joined(), _windowed([H1 + 24.0]), _coords())
    assert out2.iloc[0]["LOS_displacement"] == pytest.approx(-1.0)
    assert out2.iloc[0]["observation_timestamp"] == pd.Timestamp(DATES[1])


def test_future_observations_never_leak_into_earlier_windows() -> None:
    out = emit_group_h(_joined(), _windowed([H0]), _coords())
    row = out.iloc[0]
    assert row["LOS_displacement"] == pytest.approx(0.0), "window at H0 must not see later values"


def test_no_observation_inside_staleness_budget_yields_nan() -> None:
    """§9.1/FR-13: a staler-than-budget observation is NaN, never interpolated."""
    out = emit_group_h(_joined(), _windowed([H2 + 24.0 * 11]), _coords())  # 11 d past last obs > 10 d budget
    row = out.iloc[0]
    assert np.isnan(row["LOS_displacement"])
    assert pd.isna(row["observation_timestamp"])
    # and inside the budget (default 240 h = 10 d) it resolves
    out2 = emit_group_h(_joined(), _windowed([H2 + 24.0 * 9]), _coords())
    assert out2.iloc[0]["LOS_displacement"] == pytest.approx(-3.0)


def test_staleness_hours_grows_with_window_time() -> None:
    out = emit_group_h(_joined(), _windowed([H1, H1 + 24.0]), _coords())
    v_a = out[out.node_id == "V_A"].set_index("window_index")
    assert v_a.loc[0, "staleness_hours"] == pytest.approx(0.0)
    assert v_a.loc[1, "staleness_hours"] == pytest.approx(24.0)


def test_cumulative_displacement_referenced_to_first_observation() -> None:
    out = emit_group_h(_joined(), _windowed([H0, H2]), _coords())
    a = out[out.node_id == "V_A"].set_index("window_index")
    assert a.loc[0, "cumulative_displacement"] == pytest.approx(0.0)
    assert a.loc[1, "cumulative_displacement"] == pytest.approx(-3.0)


def test_spatial_gradient_uses_nearest_co_timed_neighbour() -> None:
    """V_A at H2: −3 mm; nearest co-timed neighbour V_B (50 m): +1 mm →
    (−3 − 1)/50 per metre."""
    out = emit_group_h(_joined(), _windowed([H2]), _coords())
    a = out[out.node_id == "V_A"].iloc[0]
    assert a["spatial_gradient"] == pytest.approx((-3.0 - 1.0) / 50.0)


def test_node_without_observation_has_nan_features_but_neighbours_stay_finite() -> None:
    win = _windowed([H2], nodes=["V_A", "V_B", "V_C"])  # V_C never observed
    out = emit_group_h(_joined(), win, _coords())
    c = out[out.node_id == "V_C"].iloc[0]
    assert np.isnan(c["LOS_displacement"])
    a = out[out.node_id == "V_A"].iloc[0]
    assert a["spatial_gradient"] == pytest.approx((-3.0 - 1.0) / 50.0), "V_C's NaN must not poison V_A"


def test_hotspot_density_counts_hot_nodes_inside_radius() -> None:
    """Default hotspot radius = 2 × 1.5 × 25 m = 75 m: V_A and V_B qualify.
    |cumulative| ≥ 10 mm marks a hotspot; push V_B over via a custom joined set."""
    out = emit_group_h(_joined(), _windowed([H2]), _coords())
    assert out.iloc[0]["local_hotspot_density"] == pytest.approx(0.0)

    hot = _joined()
    hot.loc[hot.node_id == "V_B", "insar_los_displacement_mm"] = [0.0, 6.0, 12.0]
    out_hot = emit_group_h(hot, _windowed([H2]), _coords())
    assert out_hot[out_hot.node_id == "V_A"].iloc[0]["local_hotspot_density"] == pytest.approx(1.0 / 2.0)


def test_velocity_and_coherence_carry_through() -> None:
    out = emit_group_h(_joined(), _windowed([H2]), _coords())
    a = out[out.node_id == "V_A"].iloc[0]
    assert a["LOS_velocity"] == pytest.approx(-20.0)
    assert a["coherence"] == pytest.approx(0.8)


def test_missing_window_keys_raise() -> None:
    with pytest.raises(InSARFeatureError, match="window_timestamp"):
        emit_group_h(_joined(), _windowed([H2]).drop(columns=["window_timestamp"]), _coords())


def test_missing_joined_columns_raise() -> None:
    with pytest.raises(InSARFeatureError, match="insar_los_displacement_mm"):
        emit_group_h(_joined().drop(columns=["insar_los_displacement_mm"]), _windowed([H2]), _coords())


def test_rows_align_with_window_frame() -> None:
    win = _windowed([H0, H1], nodes=["V_B", "V_A"])
    out = emit_group_h(_joined(), win, _coords())
    assert len(out) == len(win)
    assert list(out.columns[:4]) == ["event_id", "node_id", "window_index", "window_timestamp"]
    merged = win.merge(out, on=["event_id", "node_id", "window_index"], suffixes=("", "_out"))
    assert (merged["window_timestamp"] == merged["window_timestamp_out"]).all()
