"""Feature Group E — Sensor health — PRD §13 Group E, FR-4 (T-039).

§13 Group E, exactly: ``battery, RSSI, SNR, packet_loss, missing_ratio,
stuck_sensor_flag, drift_score``.

Sources (window level):
- ``battery`` / ``RSSI`` / ``SNR`` / ``packet_loss``: window means of the raw
  telemetry channels (§13 names verbatim);
- ``missing_ratio``: fraction of steps inside the window that were not
  observed on the §9.1 grid (carried through from T-029's resampling output
  when present, else 0 when every step is present in the windowed input);
- ``stuck_sensor_flag``: T-033's stuck detection re-expressed per window
  (longest near-identical run inside the window ≥ ``stuck_min_repeats``);
- ``drift_score``: T-033's trend significance restricted to the window's
  60 steps (|t|, noise-normalised), computed on the displacement channel.

Group E is what lets the models treat "the sensor is dying" differently from
"the ground is moving" — the §12 separation, in feature space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.windowing import WINDOW_TIMESTAMP
from src.preprocessing.sensor_health import (
    longest_stuck_run,
    sensor_health_params,
    trend_t_statistic,
)

__all__ = ["GROUP_E_FEATURES", "emit_group_e"]

GROUP_E_FEATURES = (
    "battery",
    "RSSI",
    "SNR",
    "packet_loss",
    "missing_ratio",
    "stuck_sensor_flag",
    "drift_score",
)

_WINDOW_KEYS = ("event_id", "node_id", "window_index", "window_timestamp", "window_start", "window_end")
_OUT_KEYS = ("event_id", "node_id", "window_index", "window_timestamp")


def emit_group_e(
    windowed: pd.DataFrame,
    raw: pd.DataFrame | None = None,
    *,
    params: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Emit Group E features per window.

    ``windowed`` is the windowing engine output (per-window telemetry
    statistics over ``battery``/``RSSI``/``SNR``/``packet_loss``). ``raw`` is
    the optional raw per-timestep frame (same event/node layout) used to
    compute ``missing_ratio``, ``stuck_sensor_flag`` and ``drift_score``
    within each window's steps; when omitted, the within-window behaviour
    features degrade to 0.0 / series-level NaN conventions.
    """
    p = params or sensor_health_params()
    required = [
        f"{c}_mean" for c in ("battery", "RSSI", "SNR", "packet_loss")
    ]
    missing = [c for c in required if c not in windowed.columns]
    if missing:
        raise ValueError(f"windowed frame missing Group E sources: {missing}")

    out = windowed[list(_WINDOW_KEYS)].copy()
    out["battery"] = windowed["battery_mean"].to_numpy(dtype=float)
    out["RSSI"] = windowed["RSSI_mean"].to_numpy(dtype=float)
    out["SNR"] = windowed["SNR_mean"].to_numpy(dtype=float)
    out["packet_loss"] = windowed["packet_loss_mean"].to_numpy(dtype=float)

    # Per-window behaviour features need the raw per-step rows. Without them,
    # missing_ratio falls back to the windowing engine's own in-window missing
    # fraction (§9 first-class data quality), and behaviour features are 0.
    if raw is None:
        if "window_missing_ratio" in windowed.columns:
            out["missing_ratio"] = windowed["window_missing_ratio"].to_numpy(dtype=float)
        else:
            out["missing_ratio"] = 0.0
        out["stuck_sensor_flag"] = 0.0
        out["drift_score"] = 0.0
        return out[list(_OUT_KEYS) + list(GROUP_E_FEATURES)]

    params_ = p
    stuck_tol = float(params_["stuck_value_tolerance"])
    stuck_min = int(params_["stuck_min_repeats"])

    # group the raw frame ONCE per series; per-window access is then O(1)
    series_groups = {
        key: g.sort_values("timestamp", kind="stable")
        for key, g in raw.groupby(["event_id", "node_id"], sort=False)
    }
    missing_col = np.zeros(len(out))
    stuck_col = np.zeros(len(out))
    drift_col = np.zeros(len(out))

    for pos, row in enumerate(out.itertuples(index=False)):
        key = (row.event_id, row.node_id)
        g = series_groups.get(key)
        if g is None:
            missing_col[pos] = stuck_col[pos] = drift_col[pos] = np.nan
            continue
        t0, t1 = float(row.window_start), float(row.window_end)
        ts = g["timestamp"].to_numpy(dtype=float)
        in_win = (ts >= t0) & (ts <= t1)
        seg = g[in_win]
        if seg.empty:
            missing_col[pos] = stuck_col[pos] = drift_col[pos] = np.nan
            continue

        window_steps = int(round((t1 - t0) / _interval_hours())) + 1
        missing_col[pos] = 1.0 - len(seg) / max(window_steps, 1)

        disp = pd.to_numeric(seg["displacement"], errors="coerce").to_numpy(dtype=float)
        stuck_col[pos] = float(longest_stuck_run(disp, stuck_tol) >= stuck_min)
        finite = disp[np.isfinite(disp)]
        if finite.size >= 3:
            sigma_diff = 1.4826 * float(np.median(np.abs(np.diff(finite) - np.median(np.diff(finite)))))
            drift_col[pos] = trend_t_statistic(finite, sigma_diff / np.sqrt(2.0) if sigma_diff > 0 else 1.0)
        else:
            drift_col[pos] = 0.0

    out["missing_ratio"] = missing_col
    out["stuck_sensor_flag"] = stuck_col
    out["drift_score"] = drift_col
    return out[list(_OUT_KEYS) + list(GROUP_E_FEATURES)]


def _interval_hours() -> float:
    from src.config import load_config

    return float(load_config("preprocessing")["resampling"]["grid_interval_minutes"]) / 60.0
