"""Packet-level data validation — PRD FR-3 + §9 (T-028).

§9: the pipeline "must tolerate missing packets, delayed packets, duplicated
packets, corrupted data, out-of-order timestamps ... treated as first-class
data-quality states, not edge cases patched in later".

This module is the pipeline's first gate after ingestion: it scans a raw node
table **per series** (``event_id`` × ``node_id`` — the §10 unit of continuous
sensing) and attaches explicit boolean flag columns for each data-quality
state. It never drops or repairs rows; downstream stages (T-029 resampling,
T-034 windowing) read the flags and decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "ValidationResult",
    "DQ_MISSING",
    "DQ_DUPLICATE",
    "DQ_OUT_OF_ORDER",
    "DQ_CORRUPTED",
    "DQ_FLAG_COLUMNS",
    "validate_packets",
]


@dataclass
class ValidationResult:
    """First-class data-quality flags, one row per input row."""

    df: pd.DataFrame
    n_missing: int
    n_duplicate: int
    n_out_of_order: int
    n_corrupted: int
    series: int = field(default=0)

    @property
    def n_flagged_rows(self) -> int:
        return int(self.df[list(DQ_FLAG_COLUMNS)].any(axis=1).sum())

    def summary(self) -> dict[str, int]:
        return {
            "missing": self.n_missing,
            "duplicate": self.n_duplicate,
            "out_of_order": self.n_out_of_order,
            "corrupted": self.n_corrupted,
            "flagged_rows": self.n_flagged_rows,
            "series": self.series,
        }


#: Data-quality flag columns added to the frame (first-class §9 states).
DQ_MISSING = "dq_missing"
DQ_DUPLICATE = "dq_duplicate"
DQ_OUT_OF_ORDER = "dq_out_of_order"
DQ_CORRUPTED = "dq_corrupted"
DQ_FLAG_COLUMNS = (DQ_MISSING, DQ_DUPLICATE, DQ_OUT_OF_ORDER, DQ_CORRUPTED)

#: Channels whose corruption is meaningful for the §11/raw schema (§8/§9 raw
#: stream). NaN in packet_loss (a count) also counts as corrupted.
_NUMERIC_CHANNELS = (
    "x",
    "y",
    "tilt_x",
    "tilt_y",
    "displacement",
    "strain",
    "vibration_rms",
    "vibration_peak",
    "battery",
    "RSSI",
    "SNR",
    "packet_loss",
)

#: Duplicate key: same series, same timestamp.
_KEY = ("event_id", "node_id", "timestamp")


def _series_groups(df: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    return [(e, n, g) for (e, n), g in df.groupby(["event_id", "node_id"], sort=False)]


def validate_packets(df: pd.DataFrame) -> ValidationResult:
    """Flag missing / duplicate / out-of-order / corrupted rows as DQ states.

    Flags (all boolean, aligned to the input row order):
      - ``dq_missing``: the (event, node, timestamp) row that the regular grid
        expects is absent (detected per series against the §8.3 10-minute
        cadence configured in configs/preprocessing.yaml).
      - ``dq_duplicate``: repeated (event, node, timestamp) keys (first
        occurrence unflagged).
      - ``dq_out_of_order``: timestamps not monotonically increasing within a
        series (first occurrence unflagged).
      - ``dq_corrupted``: non-numeric or NaN values in the raw §9 channels.
    """
    from src.config import load_config

    interval = float(
        load_config("preprocessing")["resampling"]["grid_interval_minutes"]
    ) / 60.0  # hours — timestamps in the raw table are decimal hours

    df = df.copy()
    df = df.reset_index(drop=True)  # flag arrays are positional; never trust index labels
    n = len(df)
    missing = np.zeros(n, dtype=bool)
    duplicate = np.zeros(n, dtype=bool)
    out_of_order = np.zeros(n, dtype=bool)
    corrupted = np.zeros(n, dtype=bool)

    channels = [c for c in _NUMERIC_CHANNELS if c in df.columns]
    if channels:
        num = df[channels].apply(pd.to_numeric, errors="coerce")
        corrupted = np.array(num.isna().any(axis=1).to_numpy(), dtype=bool, copy=True)
    for channel in ("node_id", "event_id"):
        if channel in df.columns:
            corrupted = corrupted | df[channel].isna().to_numpy(dtype=bool, copy=True)

    for _e, _n, g in _series_groups(df):
        idx = g.index.to_numpy()
        ts = g["timestamp"].to_numpy(dtype=float)

        # duplicates: same timestamp seen before within the series
        dup_mask = pd.Series(ts, index=g.index).duplicated(keep="first").to_numpy()
        duplicate[idx] |= dup_mask

        # out-of-order: any step backwards vs the running max of seen stamps
        run_max = np.maximum.accumulate(ts)
        backwards = np.zeros(ts.size, dtype=bool)
        backwards[1:] = ts[1:] < run_max[:-1]
        out_of_order[idx] |= backwards & ~dup_mask

        # missing rows on the regular grid: reindex the sorted unique stamps
        clean = ts[~dup_mask]
        clean = np.sort(np.unique(clean))
        if clean.size >= 1:
            t0 = clean[0]
            expected = t0 + interval * np.arange(
                int(round((clean[-1] - t0) / interval)) + 1
            )
            present = np.isin(np.rint((clean - t0) / interval), np.arange(expected.size))
            if expected.size > clean.size:
                # rows for absent grid slots — mark the *next present row* so the
                # gap itself is attributable to rows in the table
                slot = np.rint((clean - t0) / interval).astype(int)
                gaps = np.setdiff1d(np.arange(expected.size), slot, assume_unique=False)
                if gaps.size:
                    order = np.argsort(slot, kind="stable")
                    pos = np.searchsorted(slot[order], gaps, side="left")
                    hit = order[np.clip(pos, 0, order.size - 1)]
                    missing[idx[hit]] = True

    df[DQ_MISSING] = missing
    df[DQ_DUPLICATE] = duplicate
    df[DQ_OUT_OF_ORDER] = out_of_order
    df[DQ_CORRUPTED] = corrupted
    n_series = int(df.groupby(["event_id", "node_id"], sort=False).ngroups) if len(df) else 0
    return ValidationResult(
        df=df,
        n_missing=int(missing.sum()),
        n_duplicate=int(duplicate.sum()),
        n_out_of_order=int(out_of_order.sum()),
        n_corrupted=int(corrupted.sum()),
        series=n_series,
    )
