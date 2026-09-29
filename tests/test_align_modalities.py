""" acceptance tests — cross-modality temporal alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.align_modalities import (
    AlignmentError,
    align_to_windows,
    tolerance_for,
)


def _windows(n: int, start_h: float = 0.0, step_h: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"window_timestamp": start_h + step_h * np.arange(n, dtype=float)})


def test_config_tolerances_match_prd() -> None:
    assert tolerance_for("insar") == 12.0, ": InSAR ±12 hours"
    assert tolerance_for("dgps") == 1.0, ": DGPS ±1 hour"
    with pytest.raises(AlignmentError):
        tolerance_for("lidar")


def test_insar_within_twelve_hours_aligns_to_nearest_window() -> None:
    windows = _windows(3, step_h=24.0)  # InSAR-like: daily passes at 0, 24, 48 h
    obs = pd.DataFrame(
        {
            "observation_timestamp": [11.9],  # 11.9 h after the pass at 0 → within ±12
            "LOS_displacement": [-5.0],
        }
    )
    res = align_to_windows(obs, windows, modality="insar")
    assert res.n_aligned == 1
    row = res.df.iloc[0]
    assert row["window_timestamp"] == 0.0
    assert row["observation_timestamp"] == 11.9, "own observation timestamp retained"
    assert row["staleness_hours"] == pytest.approx(-11.9)


def test_dgps_beyond_one_hour_is_out_of_tolerance() -> None:
    windows = _windows(3, step_h=24.0)
    obs = pd.DataFrame({"observation_timestamp": [12.5], "DGPS_displacement": [3.0]})
    res = align_to_windows(obs, windows, modality="dgps")
    assert res.n_aligned == 0
    assert res.n_out_of_tolerance == 1
    assert res.df.empty


def test_insar_beyond_twelve_hours_is_out_of_tolerance() -> None:
    windows = _windows(2, step_h=30.0)  # passes at 0 and 30 → midpoint 15 is >12 h from both
    obs = pd.DataFrame({"observation_timestamp": [15.0]})
    res = align_to_windows(obs, windows, modality="insar")
    assert res.n_aligned == 0 and res.n_out_of_tolerance == 1


def test_boundary_exactly_at_tolerance_aligns() -> None:
    windows = _windows(2, step_h=24.0)  # windows at 0 and 24
    obs = pd.DataFrame({"observation_timestamp": [12.0]})  # exactly 12 h from both
    res = align_to_windows(obs, windows, modality="insar")
    assert res.n_aligned == 1, "tolerance is inclusive (±12 h)"


def test_nearest_window_is_chosen_not_first() -> None:
    windows = _windows(25)
    obs = pd.DataFrame({"observation_timestamp": [19.6]})
    res = align_to_windows(obs, windows, modality="dgps")
    assert res.df.iloc[0]["window_timestamp"] == 20.0


def test_staleness_is_explicit_and_signed() -> None:
    windows = _windows(25)
    obs = pd.DataFrame({"observation_timestamp": [15.6, 15.25]})
    res = align_to_windows(obs, windows, modality="dgps")
    assert "staleness_hours" in res.df.columns
    by_ts = res.df.set_index("observation_timestamp")
    assert by_ts.loc[15.6, "window_timestamp"] == 16.0
    assert by_ts.loc[15.6, "staleness_hours"] == pytest.approx(0.4)  # window ahead
    assert by_ts.loc[15.25, "window_timestamp"] == 15.0
    assert by_ts.loc[15.25, "staleness_hours"] == pytest.approx(-0.25)  # observation ahead


def test_max_staleness_tracks_worst_case() -> None:
    windows = _windows(25)
    obs = pd.DataFrame({"observation_timestamp": [0.0, 0.5]})  # 0.5 is worst case for hourly windows
    res = align_to_windows(obs, windows, modality="dgps")
    assert res.max_staleness_hours == pytest.approx(0.5)


def test_each_record_retains_its_own_payload_and_timestamp() -> None:
    windows = _windows(25)
    obs = pd.DataFrame(
        {
            "observation_timestamp": [4.0, 9.0],
            "LOS_displacement": [-1.0, -2.0],
            "coherence": [0.9, 0.8],
        }
    )
    res = align_to_windows(obs, windows, modality="insar")
    assert len(res.df) == 2
    assert list(res.df["LOS_displacement"]) == [-1.0, -2.0]
    assert list(res.df["observation_timestamp"]) == [4.0, 9.0]
    assert res.df["window_timestamp"].nunique() == 2


def test_nan_timestamps_counted_not_joined() -> None:
    windows = _windows(5)
    obs = pd.DataFrame({"observation_timestamp": [1.0, np.nan]})
    res = align_to_windows(obs, windows, modality="dgps")
    assert res.n_aligned == 1
    assert res.n_out_of_tolerance == 1


def test_empty_windows_never_align() -> None:
    obs = pd.DataFrame({"observation_timestamp": [1.0]})
    res = align_to_windows(obs, pd.DataFrame(columns=["window_timestamp"]), modality="dgps")
    assert res.n_aligned == 0
    assert res.n_out_of_tolerance == 1


def test_missing_columns_raise() -> None:
    with pytest.raises(AlignmentError):
        align_to_windows(pd.DataFrame({"t": [1.0]}), _windows(3), modality="dgps")
    with pytest.raises(AlignmentError):
        align_to_windows(
            pd.DataFrame({"observation_timestamp": [1.0]}), pd.DataFrame({"w": [1.0]}), modality="dgps"
        )
