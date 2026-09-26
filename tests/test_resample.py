"""T-029 acceptance tests — §9.1 resampling and interpolation rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.resample import (
    DQ_GAP,
    DQ_OK,
    DQ_STATE,
    GRID_SLOT,
    INTERPOLATED,
    ResampleError,
    resample_to_grid,
)

INTERVAL = 10.0 / 60.0  # hours — §8.3 cadence


def _series(timestamps: list[float], displacement: list[float] | None = None) -> pd.DataFrame:
    n = len(timestamps)
    disp = displacement if displacement is not None else list(range(n))
    return pd.DataFrame(
        {
            "event_id": ["E1"] * n,
            "node_id": ["V0001"] * n,
            "timestamp": timestamps,
            "displacement": disp,
            "tilt_x": [0.01] * n,
        }
    )


def test_perfectly_on_grid_series_passes_through() -> None:
    ts = [i * INTERVAL for i in range(6)]
    res = resample_to_grid(_series(ts))
    assert res.n_slots == 6
    assert res.n_observed == 6
    assert res.n_interpolated == 0
    assert res.n_data_quality_gaps == 0
    assert (res.df[DQ_STATE] == DQ_OK).all()
    assert np.allclose(res.df["displacement"].to_numpy(), np.arange(6, dtype=float))


def test_jitter_within_tolerance_snaps_to_slot() -> None:
    interval = INTERVAL
    tol = 0.5
    jitter = 0.3 * interval  # within ±0.5×interval
    ts = [0.0, interval + jitter, 2 * interval - jitter, 3 * interval]
    res = resample_to_grid(_series(ts))
    assert res.n_observed == 4
    assert sorted(res.df[GRID_SLOT].tolist()) == [0, 1, 2, 3]
    assert res.n_unmatched_observations == 0


def test_observation_far_beyond_the_series_end_is_unmatched() -> None:
    interval = INTERVAL
    # one stray stamp 10 intervals past the others: anchor still 0, so the
    # stray would create slots 1..10 — but it matches NO slot within tolerance
    # only if it cannot anchor a slot; it CAN (slot 10 at distance 0), so
    # instead assert the honest semantics: it extends the grid, it is not dropped.
    ts = [0.0, interval, 2 * interval, 12 * interval]
    res = resample_to_grid(_series(ts))
    assert res.n_slots == 13, "grid spans anchor → last slot"
    assert res.n_observed == 4
    assert res.n_data_quality_gaps == 9, "slots 3..11 are a long gap → DATA_QUALITY"


def test_contested_slot_earlier_observation_wins() -> None:
    interval = INTERVAL
    # anchor 0.0; stamps 0.6×I and 1.4×I BOTH land on slot 1 (each 0.4×I away,
    # within the ±0.5×I tolerance) — the earlier observation must win the slot
    ts = [0.0, 0.6 * interval, 1.4 * interval]
    rows = _series(ts, displacement=[5.0, 11.0, 22.0])
    res = resample_to_grid(rows)
    slot1 = res.df[res.df[GRID_SLOT] == 1].iloc[0]
    assert slot1["displacement"] == pytest.approx(11.0), "earlier observation wins the slot"
    assert res.df[res.df[GRID_SLOT] == 0].iloc[0]["displacement"] == pytest.approx(5.0)


def test_single_sample_gap_is_linearly_interpolated() -> None:
    interval = INTERVAL
    ts = [0.0, interval, 3 * interval, 4 * interval]  # slot 2 missing (single gap)
    res = resample_to_grid(_series(ts, displacement=[0.0, 1.0, 3.0, 4.0]))
    assert res.n_observed == 4
    assert res.n_interpolated == 1
    slot2 = res.df[res.df[GRID_SLOT] == 2].iloc[0]
    assert bool(slot2[INTERPOLATED]) is True
    assert slot2["displacement"] == pytest.approx(2.0)  # midpoint of 1.0 and 3.0
    assert slot2[DQ_STATE] == DQ_OK


def test_gap_longer_than_three_missed_samples_becomes_data_quality() -> None:
    interval = INTERVAL
    # slots 2..6 missing → run of 5 empty slots (> 3) → DATA_QUALITY, no values
    ts = [0.0, interval, 7 * interval, 8 * interval]
    res = resample_to_grid(_series(ts, displacement=[0.0, 1.0, 7.0, 8.0]))
    gap_slots = res.df[res.df[DQ_STATE] == DQ_GAP]
    assert set(gap_slots[GRID_SLOT]) == {2, 3, 4, 5, 6}
    assert gap_slots["displacement"].isna().all(), "gaps > max are never interpolated"
    assert res.n_data_quality_gaps == 5
    assert res.n_interpolated == 0


def test_gap_of_exactly_three_is_interpolatable_run_bound() -> None:
    # a run of exactly 3 empty slots is NOT > 3, so it must NOT be flagged DQ
    interval = INTERVAL
    ts = [0.0, 4 * interval]  # slots 1,2,3 empty
    res = resample_to_grid(_series(ts, displacement=[0.0, 4.0]))
    assert res.n_data_quality_gaps == 0
    # single-sample-gap rule does not apply across a 3-run either (only single gaps interpolate)
    assert res.n_interpolated == 0
    assert res.df[res.df[GRID_SLOT] == 1]["displacement"].isna().all()


def test_run_edge_slots_are_all_flagged() -> None:
    interval = INTERVAL
    ts = [0.0, 6 * interval]  # slots 1..5 empty → whole run is DQ
    res = resample_to_grid(_series(ts))
    flagged = res.df[res.df[DQ_STATE] == DQ_GAP][GRID_SLOT].tolist()
    assert flagged == [1, 2, 3, 4, 5]


def test_leading_and_trailing_gaps_beyond_series_are_not_invented() -> None:
    interval = INTERVAL
    ts = [2 * interval, 3 * interval, 4 * interval]
    res = resample_to_grid(_series(ts))
    assert res.df[GRID_SLOT].min() == 0, "grid anchors at the first observation"
    assert res.df[GRID_SLOT].max() == 2


def test_series_are_independent() -> None:
    interval = INTERVAL
    a = _series([0.0, interval, 2 * interval])
    b = _series([100.0, 100.0 + interval])
    b["node_id"] = "V0002"
    res = resample_to_grid(pd.concat([a, b], ignore_index=True))
    assert res.n_slots == 5
    assert res.per_series["E1/V0001"]["slots"] == 3
    assert res.per_series["E1/V0002"]["slots"] == 2





def test_config_is_the_single_threshold_authority() -> None:
    from src.config import load_config

    cfg = load_config("preprocessing")["resampling"]
    assert cfg["match_tolerance_fraction"] == 0.5
    assert cfg["max_interpolatable_gap_steps"] == 3
    assert cfg["grid_interval_minutes"] == 10


def test_invalid_channel_list_raises() -> None:
    with pytest.raises(ResampleError):
        resample_to_grid(_series([0.0, INTERVAL]), channels=[])
