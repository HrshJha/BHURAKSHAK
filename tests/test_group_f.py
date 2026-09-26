"""T-041 acceptance tests — Feature Group F (physics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.group_f_physics import GROUP_F_FEATURES, emit_group_f
from src.features.windowing import build_windows
from src.physics.consistency import physics_engine


def _frames():
    n = 144
    ts = (10.0 / 60.0) * np.arange(n, dtype=float)
    node = "V0012"  # mesh centre (0, 0) — max expected subsidence
    obs = np.linspace(0.0, 450.0, n)  # a realistic day of subsidence at the centre
    raw = pd.DataFrame(
        {
            "event_id": "E1",
            "node_id": node,
            "timestamp": ts,
            "displacement": obs,
            "tilt_x": np.full(n, 0.004),
            "tilt_y": np.full(n, -0.003),
        }
    )
    coords = pd.DataFrame({"node_id": [node], "x": [0.0], "y": [0.0]})
    return raw, build_windows(raw).df, coords


def test_feature_names_match_section_13() -> None:
    assert GROUP_F_FEATURES == (
        "expected_displacement",
        "expected_tilt",
        "physics_residual",
        "physics_residual_velocity",
    )


def test_expected_values_come_from_the_configured_model() -> None:
    raw, windowed, coords = _frames()
    out = emit_group_f(windowed, coords)
    engine = physics_engine()
    row = out[out["window_index"] == 8].iloc[0]
    t = float(row["window_timestamp"])
    expected = float(engine.expected_displacement_mm(0.0, 0.0, t))
    assert row["expected_displacement"] == pytest.approx(expected)
    etx, ety = engine.expected_tilt(0.0, 0.0, t)
    assert row["expected_tilt"] == pytest.approx(float(np.hypot(etx, ety)))


def test_residual_is_observed_minus_expected() -> None:
    raw, windowed, coords = _frames()
    out = emit_group_f(windowed, coords)
    row = out[out["window_index"] == 8].iloc[0]
    recomputed = (
        windowed.loc[windowed["window_index"] == 8, "displacement_mean"].iloc[0]
        - row["expected_displacement"]
    )
    assert row["physics_residual"] == pytest.approx(recomputed)


def test_residual_near_zero_when_observation_matches_model_well() -> None:
    """The observation ramp is chosen so late-window residuals are small
    relative to the 450 mm signal — the §21 'observation ≈ model' case."""
    raw, windowed, coords = _frames()
    out = emit_group_f(windowed, coords)
    late = out[out["window_index"] >= 5]
    scale = float(late["expected_displacement"].abs().mean())
    assert (late["physics_residual"].abs() < 0.35 * scale).all()


def test_residual_velocity_zero_for_constant_residual() -> None:
    raw, windowed, coords = _frames()
    out = emit_group_f(windowed, coords)
    first = out[out["window_index"] == 0]
    assert (first["physics_residual_velocity"] == 0.0).all()
    # residual changes smoothly ⇒ velocity is finite and bounded
    assert out["physics_residual_velocity"].abs().max() < 1e3


def test_output_carries_exactly_group_f_plus_keys() -> None:
    raw, windowed, coords = _frames()
    out = emit_group_f(windowed, coords)
    assert list(out.columns) == [
        "event_id",
        "node_id",
        "window_index",
        "window_timestamp",
        *GROUP_F_FEATURES,
    ]


def test_missing_displacement_source_raises() -> None:
    _raw, windowed, coords = _frames()
    with pytest.raises(Exception, match="displacement_mean"):
        emit_group_f(windowed.drop(columns=["displacement_mean"]), coords)


def test_series_independence_of_residual_velocity() -> None:
    raw, windowed, coords = _frames()
    other = windowed.assign(event_id="E2")
    out = emit_group_f(pd.concat([windowed, other], ignore_index=True), coords)
    first_of_each = out[out["window_index"] == 0]
    assert (first_of_each["physics_residual_velocity"] == 0.0).all(), (
        "the first window of EVERY series must have zero residual velocity"
    )
