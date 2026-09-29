"""Create ordered, fixed-length windows from per-node sensor series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import load_config

__all__ = [
    "WindowingError",
    "WindowResult",
    "windowing_params",
    "build_windows",
    "assert_windowed",
]

WINDOW_TIMESTAMP = "window_timestamp"
WINDOW_START = "window_start"
WINDOW_END = "window_end"
WINDOW_INDEX = "window_index"


class WindowingError(ValueError):
    """Raised on invalid windowing inputs or raw frames reaching model inputs."""


@dataclass
class WindowResult:
    df: pd.DataFrame  # one row per window
    n_windows: int
    window_steps: int
    stride: int
    series: int

    def summary(self) -> dict[str, int]:
        return {
            "windows": self.n_windows,
            "window_steps": self.window_steps,
            "stride": self.stride,
            "series": self.series,
        }


def windowing_params() -> tuple[int, int]:
    cfg = load_config("preprocessing")["windowing"]
    window = int(cfg["window_steps"])
    stride = int(cfg["stride"])
    if window < 2:
        raise WindowingError("window_steps must be >= 2")
    if stride < 1:
        raise WindowingError("stride must be >= 1")
    return window, stride


def _slope(values: np.ndarray) -> float:
    """OLS slope of a window (per step), fitted over FINITE samples only.: missing steps (e.g. DROPOUT runs) are first-class, not errors —
 the regression simply skips them. Returns NaN only when fewer than two
 finite samples exist (a window that cannot support a slope at all).
 """
    ok = np.isfinite(values)
    n = int(ok.sum())
    if n < 2:
        return float("nan")
    t = np.arange(values.size, dtype=float)[ok]
    v = values[ok]
    t_mean = t.mean()
    denom = float(((t - t_mean) ** 2).sum())
    if denom <= 0:
        return 0.0
    return float(((t - t_mean) * (v - v.mean())).sum() / denom)


def _velocity(values: np.ndarray, interval_hours: float) -> float:
    """Mean per-hour change across the window: (last − first finite) / duration.

 The duration spans the full window; only the endpoints must be finite.
 Returns NaN when fewer than two finite samples exist.
 """
    ok = np.flatnonzero(np.isfinite(values))
    if ok.size < 2 or interval_hours <= 0:
        return float("nan") if ok.size < 2 else 0.0
    duration_hours = (values.size - 1) * interval_hours
    return float((values[ok[-1]] - values[ok[0]]) / duration_hours)


def _acceleration(velocities: np.ndarray, interval_hours: float) -> float:
    """Per-hour² change of the per-step velocity proxy inside the window.

 Velocity samples are the window's finite per-step deltas; acceleration is
 the change between the first and last usable velocity sample over the
 elapsed span. Zero when fewer than two usable deltas exist (: gaps are
 skipped, never invented).
 """
    if velocities.size < 3 or interval_hours <= 0:
        return 0.0
    finite_pairs = np.isfinite(velocities[1:]) & np.isfinite(velocities[:-1])
    positions = np.flatnonzero(finite_pairs)
    if positions.size < 2:
        return 0.0
    deltas = (velocities[1:] - velocities[:-1])[finite_pairs]
    v_first = deltas[0] / interval_hours
    v_last = deltas[-1] / interval_hours
    span_steps = int(positions[-1] - positions[0])
    if span_steps == 0:
        return 0.0
    return float((v_last - v_first) / (span_steps * interval_hours))


def build_windows(
    df: pd.DataFrame,
    *,
    channels: tuple[str, ...] | None = None,
    labels: tuple[str, ...] = ("anomaly_label", "risk_label", "progression_label", "fault_label"),
) -> WindowResult:
    """Build windows over each (event_id, node_id) series.

 One output row per window per series, carrying ``window_index``, window
 start/end timestamps, per-channel statistics (``<ch>_mean`` / ``_std`` /
 ``_slope`` / ``_velocity`` / ``_acceleration``) and the first label value
 seen in the window (windows are within one event, so labels are
 internally consistent; majority semantics are left to ).
 """
    window, stride = windowing_params()
    interval_hours = float(
        load_config("preprocessing")["resampling"]["grid_interval_minutes"]
    ) / 60.0

    if channels is None:
        channels = tuple(
            c
            for c in (
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
            if c in df.columns
        )
    if not channels:
        raise WindowingError("no value channels to window over")

    rows: list[dict[str, object]] = []
    n_series = 0
    for (event, node), g in df.groupby(["event_id", "node_id"], sort=False):
        g = g.sort_values("timestamp", kind="stable")
        n_series += 1
        n = len(g)
        if n < window:
            continue  # a series shorter than one window produces no windows
        ts = g["timestamp"].to_numpy(dtype=float)
        series_data: dict[str, np.ndarray] = {}
        for ch in channels:
            series_data[ch] = pd.to_numeric(g[ch], errors="coerce").to_numpy(dtype=float)
        label_data = {col: (g[col].to_numpy() if col in g.columns else None) for col in labels}

        w_idx = 0
        start = 0
        while start + window <= n:
            end = start + window  # exclusive
            row: dict[str, object] = {
                "event_id": event,
                "node_id": node,
                WINDOW_INDEX: w_idx,
                WINDOW_START: float(ts[start]),
                WINDOW_END: float(ts[end - 1]),
                WINDOW_TIMESTAMP: float(ts[end - 1]),  # window anchored at its last step
            }
            for ch in channels:
                v = series_data[ch][start:end]
                finite = int(np.isfinite(v).sum())
                row[f"{ch}_mean"] = float(np.nanmean(v)) if finite else float("nan")
                row[f"{ch}_std"] = float(np.nanstd(v))
                row[f"{ch}_min"] = float(np.nanmin(v)) if finite else float("nan")
                row[f"{ch}_max"] = float(np.nanmax(v)) if finite else float("nan")
                row[f"{ch}_slope"] = _slope(v)
                row[f"{ch}_velocity"] = _velocity(v, interval_hours)
                row[f"{ch}_acceleration"] = _acceleration(v, interval_hours)
            # first-class data quality: fraction of steps with ANY missing
            # core channel in this window (Group E's missing_ratio carries it)
            core = np.column_stack([series_data[ch][start:end] for ch in channels])
            row["window_missing_ratio"] = float((~np.isfinite(core)).any(axis=1).mean())
            for col, accessor in label_data.items():
                if accessor is not None:
                    row[col] = accessor[start]  # first label in the window
            rows.append(row)
            w_idx += 1
            start += stride

    out = pd.DataFrame(rows)
    return WindowResult(
        df=out,
        n_windows=len(out),
        window_steps=window,
        stride=stride,
        series=n_series,
    )


def assert_windowed(df: pd.DataFrame) -> None:
    """Guard: raise if a frame looks like raw per-timestep rows.

 Models must consume window-level features. A frame that carries the raw
 ``timestamp`` column without the windowing markers (``window_index`` /
 ``window_timestamp``) is raw — pass it through:func:`build_windows` first.
 """
    if WINDOW_INDEX not in df.columns or WINDOW_TIMESTAMP not in df.columns:
        raise WindowingError(
            "raw per-timestep rows detected (missing 'window_index'/'window_timestamp'); "
            "models must consume windows — run src.features.windowing.build_windows first"
        )
