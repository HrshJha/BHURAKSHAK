"""T-035 acceptance tests — Feature Group A (physical)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.group_a_physical import GROUP_A_FEATURES, GROUP_A_SOURCES, emit_group_a
from src.features.windowing import build_windows


def _windowed() -> pd.DataFrame:
    n = 144
    ts = (10.0 / 60.0) * np.arange(n)
    df = pd.DataFrame(
        {
            "event_id": "E1",
            "node_id": "V0001",
            "timestamp": ts,
            "tilt_x": 0.01 * np.ones(n),
            "tilt_y": -0.02 * np.ones(n),
            "tilt_magnitude": 0.0224 * np.ones(n),
            "displacement": np.linspace(0.0, 12.0, n),
            "strain": 1e-4 * np.ones(n),
            "risk_label": "NORMAL",
        }
    )
    return build_windows(df).df


def test_feature_names_match_section_13_exactly() -> None:
    assert GROUP_A_FEATURES == ("tilt_x", "tilt_y", "tilt_magnitude", "displacement", "strain")


def test_emit_produces_exactly_group_a_columns_plus_keys() -> None:
    out = emit_group_a(_windowed())
    assert set(GROUP_A_FEATURES) <= set(out.columns)
    expected_keys = {"event_id", "node_id", "window_index", "window_timestamp"}
    assert set(out.columns) == set(GROUP_A_FEATURES) | expected_keys


def test_values_are_window_means_of_raw_channels() -> None:
    windowed = _windowed()
    out = emit_group_a(windowed)
    assert out["tilt_x"].iloc[0] == pytest.approx(0.01)
    assert out["tilt_y"].iloc[0] == pytest.approx(-0.02)
    # displacement ramps 0→12 mm over the day; window 0 mean ≈ mean of first 60 steps
    expected = np.linspace(0.0, 12.0, 144)[:60].mean()
    assert out["displacement"].iloc[0] == pytest.approx(expected)


def test_one_row_per_window_preserved() -> None:
    windowed = _windowed()
    out = emit_group_a(windowed)
    assert len(out) == len(windowed) == 9


def test_missing_source_channel_raises_loudly() -> None:
    windowed = _windowed().drop(columns=["strain_mean"])
    with pytest.raises(ValueError, match="strain_mean"):
        emit_group_a(windowed)


def test_sources_mapping_is_one_to_one() -> None:
    assert len(GROUP_A_SOURCES) == len(GROUP_A_FEATURES)
    assert set(GROUP_A_SOURCES) == set(GROUP_A_FEATURES)
