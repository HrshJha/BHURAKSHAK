"""§9.1 resampling and interpolation rule (T-029).

PRD §9.1: "feature windows are built on a fixed grid (default:
nearest-neighbor within one half of the sampling interval, linear interpolation for
single-sample gaps, no interpolation across gaps longer than 3 missed samples
— those become ``DATA_QUALITY`` flags instead)".

This module realises exactly that rule, config-driven from
configs/preprocessing.yaml (``resampling`` block):

- observations are snapped onto a regular grid anchored at the first
  observation of each series (``event_id`` × ``node_id``);
- an observation joins the grid slot only if it falls within
  ±``match_tolerance_fraction`` × interval of that slot (nearest-neighbour
  match); observations outside every slot's tolerance are dropped as junk and
  counted;
- slots with exactly one matched observation take that value;
- slots whose neighbourhood has a single-sample gap on one side get a linear
  interpolation across that single gap;
- runs of more than ``max_interpolatable_gap_steps`` consecutive empty slots
  are NEVER interpolated — they are emitted as ``DATA_QUALITY`` flag rows so
  downstream stages can see the outage explicitly (§9: first-class states).

The output is a per-slot tidy table: one row per (series, grid slot) with
value columns, an ``interpolated`` flag, and a ``data_quality`` column.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import load_config

__all__ = [
    "ResampleError",
    "ResampleResult",
    "resample_to_grid",
    "GRID_SLOT",
    "INTERPOLATED",
    "DQ_STATE",
]

GRID_SLOT = "grid_slot"
INTERPOLATED = "interpolated"
DQ_STATE = "data_quality"

#: §9.1 / §10 data-quality vocabulary for resampled slots.
DQ_OK = "OK"
DQ_GAP = "DATA_QUALITY"

_VALUE_CHANNELS = (
    "tilt_x",
    "tilt_y",
    "tilt_magnitude",
    "displacement",
    "strain",
    "vibration_rms",
    "vibration_peak",
    "battery",
    "RSSI",
    "SNR",
    "packet_loss",
)


class ResampleError(ValueError):
    """Raised on invalid resampling input or configuration."""


@dataclass
class ResampleResult:
    """Resampled grid table plus audit counters (nothing silently absorbed)."""

    df: pd.DataFrame
    n_slots: int
    n_observed: int
    n_interpolated: int
    n_data_quality_gaps: int
    n_unmatched_observations: int = 0
    per_series: dict[str, dict[str, int]] = field(default_factory=dict)

    def summary(self) -> dict[str, int]:
        return {
            "slots": self.n_slots,
            "observed": self.n_observed,
            "interpolated": self.n_interpolated,
            "data_quality_gaps": self.n_data_quality_gaps,
            "unmatched_observations": self.n_unmatched_observations,
        }


def _resampling_params() -> tuple[float, float, int]:
    cfg = load_config("preprocessing")["resampling"]
    interval = float(cfg["grid_interval_minutes"]) / 60.0  # hours (raw timestamps)
    tol = float(cfg["match_tolerance_fraction"])
    max_gap = int(cfg["max_interpolatable_gap_steps"])
    if interval <= 0:
        raise ResampleError("grid_interval_minutes must be > 0")
    if not 0 < tol < 1:
        raise ResampleError("match_tolerance_fraction must be in (0, 1)")
    if max_gap < 1:
        raise ResampleError("max_interpolatable_gap_steps must be >= 1")
    return interval, tol, max_gap


def _match_to_grid(ts: np.ndarray, t0: float, interval: float, tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour snap of observations onto grid slots.

    Returns ``(slot_index, matched_obs_position, unmatched_obs_position)``.
    An observation matches the nearest slot only if |t_obs − t_slot| ≤ tol·interval.
    A slot keeps at most one observation — its nearest; others are unmatched.
    """
    slot_of = np.rint((ts - t0) / interval).astype(int)
    slot_time = t0 + slot_of * interval
    within_tol = np.abs(ts - slot_time) <= tol * interval
    order = np.argsort(ts, kind="stable")  # earlier observation wins a contested slot
    matched_pos: dict[int, int] = {}
    for pos in order:
        if within_tol[pos] and slot_of[pos] not in matched_pos:
            matched_pos[int(slot_of[pos])] = int(pos)
    slots = np.array(sorted(matched_pos), dtype=int)
    matched_obs = np.array([matched_pos[s] for s in slots], dtype=int) if slots.size else np.empty(0, dtype=int)
    unmatched_obs = np.flatnonzero(~np.isin(np.arange(ts.size), matched_obs))
    return slots, matched_obs, unmatched_obs


def _linear_fill(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation across single-sample gaps only (never runs)."""
    out = values.copy()
    n = valid.size
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        run = j - i
        if run == 1 and 0 < i and j < n:  # single-sample gap with both neighbours
            left, right = values[i - 1], values[j]
            out[i] = left + (right - left) * (1.0 / 2.0)
            valid[i] = True
        i = j if j > i else i + 1
    return out, valid


def _resample_series(
    g: pd.DataFrame,
    interval: float,
    tol: float,
    max_gap: int,
    channels: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    ts_all = pd.to_numeric(g["timestamp"], errors="coerce").to_numpy(dtype=float)
    good = ~pd.isna(ts_all)
    ts = ts_all[good]
    positions = np.flatnonzero(good)  # g-row position of every entry in ts
    if ts.size == 0:
        return pd.DataFrame(columns=[GRID_SLOT, "timestamp", *channels, INTERPOLATED, DQ_STATE]), {
            "slots": 0,
            "observed": 0,
            "interpolated": 0,
            "data_quality_gaps": 0,
            "unmatched": int(good.size - ts.size),
        }

    t0 = float(ts.min())
    slots, matched_obs, unmatched_obs = _match_to_grid(ts, t0, interval, tol)
    n_unmatched = int(unmatched_obs.size + (good.size - ts.size))  # junk timestamps count too

    last = int(np.rint((ts.max() - t0) / interval))
    all_slots = np.arange(0, last + 1, dtype=int)
    values = np.full((all_slots.size, len(channels)), np.nan)
    obs_at_slot = np.zeros(all_slots.size, dtype=bool)

    pos_of_slot = {int(s): k for k, s in enumerate(slots)}
    for slot, pos in pos_of_slot.items():
        row = g.iloc[positions[matched_obs[pos]]]
        for k, ch in enumerate(channels):
            if ch in g.columns:
                values[slot, k] = pd.to_numeric(pd.Series([row[ch]]), errors="coerce").iloc[0]
        obs_at_slot[slot] = True

    interpolated = np.zeros(all_slots.size, dtype=bool)
    for k, ch in enumerate(channels):
        valid = obs_at_slot & ~np.isnan(values[:, k])
        if valid.any():
            filled, valid_k = _linear_fill(values[:, k], valid.copy())
            interpolated |= valid_k & ~obs_at_slot
            values[:, k] = filled

    # runs of empty slots longer than max_gap are DATA_QUALITY, never filled
    gap_run = np.zeros(all_slots.size, dtype=int)
    run = 0
    for i in range(all_slots.size):
        run = run + 1 if not obs_at_slot[i] else 0
        gap_run[i] = run
    dq = np.zeros(all_slots.size, dtype=bool)
    run = 0
    for i in range(all_slots.size - 1, -1, -1):
        run = run + 1 if not obs_at_slot[i] else 0
        if not obs_at_slot[i] and max(gap_run[i], run) > max_gap:
            dq[i] = True
    # any slot inside a >max_gap run is DATA_QUALITY (whole run, not just its tail)
    above = gap_run > max_gap
    if above.any():
        ends = np.flatnonzero(above & ~np.concatenate([above[1:], [False]]))
        for end in ends:
            start = end - gap_run[end] + 1
            dq[start : end + 1] = True
    # interpolated values may not live inside a DATA_QUALITY run
    values[dq] = np.nan
    interpolated &= ~dq

    out = pd.DataFrame(
        {
            GRID_SLOT: all_slots,
            "timestamp": t0 + all_slots * interval,
            INTERPOLATED: interpolated,
            DQ_STATE: np.where(dq, DQ_GAP, DQ_OK),
        }
    )
    for k, ch in enumerate(channels):
        out[ch] = values[:, k]
    stats = {
        "slots": int(all_slots.size),
        "observed": int(obs_at_slot.sum()),
        "interpolated": int(interpolated.sum()),
        "data_quality_gaps": int(dq.sum()),
        "unmatched": n_unmatched,
    }
    return out, stats


def resample_to_grid(df: pd.DataFrame, *, channels: list[str] | None = None) -> ResampleResult:
    """Resample raw node rows onto the §9.1 fixed grid, per (event, node) series.

    One output row per (series, grid slot). Slots carry ``data_quality ==
    "DATA_QUALITY"`` when they sit inside a gap longer than the configured
    maximum; those slots are never interpolated.
    """
    interval, tol, max_gap = _resampling_params()
    if channels is None:
        use_channels = [c for c in _VALUE_CHANNELS if c in df.columns]
    else:
        use_channels = list(channels)  # an explicit empty list is a caller error
    if not use_channels:
        raise ResampleError("no value channels to resample")

    frames: list[pd.DataFrame] = []
    per_series: dict[str, dict[str, int]] = {}
    totals = {"slots": 0, "observed": 0, "interpolated": 0, "data_quality_gaps": 0, "unmatched": 0}
    for (event, node), g in df.groupby(["event_id", "node_id"], sort=False):
        out, stats = _resample_series(g, interval, tol, max_gap, use_channels)
        out.insert(0, "node_id", node)
        out.insert(0, "event_id", event)
        frames.append(out)
        per_series[f"{event}/{node}"] = stats
        for key in totals:
            totals[key] += stats.get(key if key != "unmatched_observations" else "unmatched", 0)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return ResampleResult(
        df=result,
        n_slots=totals["slots"],
        n_observed=totals["observed"],
        n_interpolated=totals["interpolated"],
        n_data_quality_gaps=totals["data_quality_gaps"],
        n_unmatched_observations=totals["unmatched"],
        per_series=per_series,
    )
