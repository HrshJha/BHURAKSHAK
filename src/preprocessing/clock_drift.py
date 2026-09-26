"""Node clock-drift estimation and correction — PRD §9.1 (T-030).

§9.1: "each uplink packet carries the node's local timestamp; the gateway also
stamps its own receipt time. The two are compared per node to estimate drift,
and a drift-correction offset is applied during feature generation rather than
trusting raw node timestamps indefinitely — this offset is logged, not
silently absorbed."

Design:
- ``estimate_clock_drift`` consumes per-node (node-stamp, receipt-stamp)
  pairs and produces a robust per-node offset estimate (median of the
  differences), plus an informational drift-rate (seconds per hour) from a
  linear fit.
- Nodes with fewer than ``clock_drift.min_samples_for_estimate`` pairs are
  never corrected (flag ``INSUFFICIENT_SAMPLES``); nodes whose estimated
  offset exceeds ``clock_drift.max_abs_offset_seconds`` in magnitude are
  flagged and NOT corrected (the offset is treated as suspect, not absorbed).
- ``apply_clock_correction`` shifts node timestamps by the estimated offset at
  feature-generation time and returns the applied-correction log — every
  applied offset is visible in ``drift_log`` output, never silently absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import load_config

__all__ = [
    "ClockDriftError",
    "ClockDriftEstimate",
    "estimate_clock_drift",
    "apply_clock_correction",
    "drift_log",
]

FLAG_OK = ""
FLAG_INSUFFICIENT = "INSUFFICIENT_SAMPLES"
FLAG_SUSPECT = "OFFSET_EXCEEDS_MAX_NOT_APPLIED"


class ClockDriftError(ValueError):
    """Raised on invalid clock-drift inputs or configuration."""


@dataclass(frozen=True)
class ClockDriftEstimate:
    node_id: str
    n_samples: int
    offset_seconds: float  # node clock minus gateway clock (what must be subtracted)
    drift_rate_seconds_per_hour: float
    corrected: bool
    flag: str


def _params() -> tuple[int, float]:
    cfg = load_config("preprocessing")["clock_drift"]
    min_n = int(cfg["min_samples_for_estimate"])
    max_off = float(cfg["max_abs_offset_seconds"])
    if min_n < 2:
        raise ClockDriftError("min_samples_for_estimate must be >= 2")
    if max_off <= 0:
        raise ClockDriftError("max_abs_offset_seconds must be > 0")
    return min_n, max_off


def estimate_clock_drift(receipts: pd.DataFrame) -> list[ClockDriftEstimate]:
    """Estimate per-node drift from node-stamp vs gateway receipt-stamp pairs.

    ``receipts`` must carry ``node_id``, ``node_timestamp`` and
    ``receipt_timestamp`` (same unit — seconds). The offset is the median of
    (receipt − node) differences: robust against packet jitter and outliers.
    """
    min_n, max_off = _params()
    for col in ("node_id", "node_timestamp", "receipt_timestamp"):
        if col not in receipts.columns:
            raise ClockDriftError(f"receipts must carry column {col!r}")

    estimates: list[ClockDriftEstimate] = []
    for node, g in receipts.groupby("node_id", sort=True):
        node_ts = pd.to_numeric(g["node_timestamp"], errors="coerce").to_numpy(dtype=float)
        rcpt_ts = pd.to_numeric(g["receipt_timestamp"], errors="coerce").to_numpy(dtype=float)
        ok = ~(np.isnan(node_ts) | np.isnan(rcpt_ts))
        node_ts, rcpt_ts = node_ts[ok], rcpt_ts[ok]
        n = int(node_ts.size)
        if n < min_n:
            estimates.append(
                ClockDriftEstimate(str(node), n, float("nan"), float("nan"), False, FLAG_INSUFFICIENT)
            )
            continue
        delta = rcpt_ts - node_ts
        offset = float(np.median(delta))
        # informational drift rate: does the offset grow with receipt time?
        if np.ptp(rcpt_ts) > 0:
            rate = float(np.polyfit(rcpt_ts, delta, 1)[0])
        else:
            rate = 0.0
        corrected = abs(offset) <= max_off
        estimates.append(
            ClockDriftEstimate(
                node_id=str(node),
                n_samples=n,
                offset_seconds=offset,
                drift_rate_seconds_per_hour=rate * 3600.0,
                corrected=corrected,
                flag=FLAG_OK if corrected else FLAG_SUSPECT,
            )
        )
    return estimates


def drift_log(estimates: list[ClockDriftEstimate]) -> pd.DataFrame:
    """The mandatory §9.1 log: one row per node, corrected or not."""
    return pd.DataFrame(
        [
            {
                "node_id": e.node_id,
                "n_samples": e.n_samples,
                "offset_seconds": e.offset_seconds,
                "drift_rate_seconds_per_hour": e.drift_rate_seconds_per_hour,
                "corrected": e.corrected,
                "flag": e.flag,
            }
            for e in estimates
        ]
    )


def apply_clock_correction(
    df: pd.DataFrame,
    estimates: list[ClockDriftEstimate],
    *,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, list[str]]:
    """Apply the per-node offset at feature-generation time.

    Timestamps in ``df`` are decimal hours (raw node-table convention); the
    offsets are seconds. ``corrected`` nodes get
    ``t → t − offset/3600`` so node stamps land on gateway time. Nodes
    without a usable estimate are left untouched and reported. The input
    frame is never mutated.
    """
    mapping = {e.node_id: e.offset_seconds for e in estimates if e.corrected}
    out = df.copy()
    if timestamp_col not in out.columns:
        raise ClockDriftError(f"missing timestamp column {timestamp_col!r}")
    offsets_s = out["node_id"].map(mapping).astype(float).fillna(0.0).to_numpy()
    hours = pd.to_numeric(out[timestamp_col], errors="coerce").to_numpy(dtype=float)
    out[timestamp_col] = hours - offsets_s / 3600.0
    applied = sorted(mapping)
    return out, applied
