""" acceptance tests — packet-level data validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.validation import (
    DQ_CORRUPTED,
    DQ_DUPLICATE,
    DQ_MISSING,
    DQ_OUT_OF_ORDER,
    validate_packets,
)

INTERVAL = 10.0 / 60.0  # grid interval in hours (timestamps are decimal hours)


def _base_rows() -> list[dict]:
    return [
        {
            "event_id": "E1",
            "node_id": "V0001",
            "timestamp": t,
            "tilt_x": 0.01,
            "displacement": 1.0,
            "battery": 4.0,
        }
        for t in (0.0, INTERVAL, 2 * INTERVAL, 3 * INTERVAL, 4 * INTERVAL)
    ]


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_clean_series_has_no_flags() -> None:
    res = validate_packets(_frame(_base_rows()))
    assert not res.df[DQ_MISSING].any()
    assert not res.df[DQ_DUPLICATE].any()
    assert not res.df[DQ_OUT_OF_ORDER].any()
    assert not res.df[DQ_CORRUPTED].any()
    assert res.summary()["series"] == 1


def test_missing_row_detected_on_grid() -> None:
    rows = _base_rows()
    del rows[2]  # drop t = 2×interval
    res = validate_packets(_frame(rows))
    assert res.n_missing == 1
    flagged_ts = res.df.loc[res.df[DQ_MISSING], "timestamp"]
    # the flag lands on the next present row after the gap
    assert float(flagged_ts.iloc[0]) == 3 * INTERVAL


def test_duplicate_timestamp_flagged() -> None:
    rows = _base_rows()
    dup = dict(rows[1])
    dup["displacement"] = 99.0  # conflicting payload, same key
    rows.insert(2, dup)
    res = validate_packets(_frame(rows))
    assert res.n_duplicate == 1
    assert float(res.df.loc[res.df[DQ_DUPLICATE], "timestamp"].iloc[0]) == INTERVAL


def test_out_of_order_flagged() -> None:
    rows = _base_rows()
    rows[1], rows[2] = rows[2], rows[1]
    res = validate_packets(_frame(rows))
    assert res.n_out_of_order >= 1
    assert res.df[DQ_OUT_OF_ORDER].any()


def test_nan_payload_is_corrupted() -> None:
    rows = _base_rows()
    rows[3]["tilt_x"] = np.nan
    res = validate_packets(_frame(rows))
    assert res.n_corrupted == 1
    assert bool(res.df.loc[3, DQ_CORRUPTED]) is True


def test_non_numeric_payload_is_corrupted() -> None:
    rows = _base_rows()
    rows[0]["displacement"] = "garbage"
    res = validate_packets(_frame(rows))
    assert res.n_corrupted == 1


def test_series_are_independent() -> None:
    rows = _base_rows()
    rows_b = [dict(r, node_id="V0002") for r in _base_rows()]
    del rows_b[1]  # gap only in node B
    res = validate_packets(_frame(rows + rows_b))
    assert res.summary()["series"] == 2
    assert res.n_missing == 1
    flagged = res.df.loc[res.df[DQ_MISSING]]
    assert (flagged["node_id"] == "V0002").all()


def test_events_are_independent() -> None:
    rows = _base_rows()
    # a new event may start its own grid far from event E1's timeline
    rows_c = [dict(r, event_id="E2", timestamp=r["timestamp"] + 1000.0) for r in _base_rows()]
    res = validate_packets(_frame(rows + rows_c))
    assert res.n_out_of_order == 0, "timestamps of different events must not be compared"


def test_flags_never_modify_or_drop_rows() -> None:
    df_in = _frame(_base_rows())
    res = validate_packets(df_in)
    assert len(res.df) == len(df_in)
    assert (res.df["displacement"].to_numpy() == df_in["displacement"].to_numpy()).all()


def test_multiple_states_can_coexist_on_one_row() -> None:
    rows = _base_rows()
    dup = dict(rows[1])
    dup["battery"] = np.nan
    rows.insert(2, dup)
    res = validate_packets(_frame(rows))
    dup_rows = res.df[res.df[DQ_DUPLICATE]]
    assert bool(dup_rows[DQ_CORRUPTED].any()), "duplicate row may also be corrupted"


def test_validator_handles_read_only_series_views(monkeypatch) -> None:
    df = _frame(_base_rows())
    original = pd.Series.to_numpy

    def readonly_to_numpy(series, *args, **kwargs):
        values = original(series, *args, **kwargs)
        if isinstance(values, np.ndarray):
            values.setflags(write=False)
        return values

    monkeypatch.setattr(pd.Series, "to_numpy", readonly_to_numpy)
    result = validate_packets(df)
    assert result.n_corrupted == 0
    assert len(result.df) == len(df)


def test_validator_accepts_arrow_backed_copy_on_write_frame() -> None:
    frame = _frame(_base_rows()).convert_dtypes(dtype_backend="pyarrow")
    with pd.option_context("mode.copy_on_write", True):
        result = validate_packets(frame)
    assert result.n_corrupted == 0
    assert len(result.df) == len(frame)
