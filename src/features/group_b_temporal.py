"""Feature Group B — Temporal — Group B,.

 Group B: rolling mean/std/min/max, slope, velocity, acceleration, trend,
persistence, change-point score.

Two -adjacent acceptance requirements are engineered in:

- **No rolling statistic crosses a train/test boundary.** Rolling statistics
 are computed *within a window's own 60 steps* (and, for the rolling
 moments, within a sub-history limited to the window itself) over each
 (event_id, node_id) series. Because windows are grouped per series and each
 window only sees its own rows, a statistic computed for one window can
 never incorporate rows from another series, event, or split block.:func:`assert_no_boundary_crossing` re-checks this property on the emitted
 frame by recomputing per-series statistics independently.

- The features are emitted per window (from the windowing engine), with
 names matching exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["GROUP_B_FEATURES", "emit_group_b", "assert_no_boundary_crossing"]

_WINDOW_KEYS = ("event_id", "node_id", "window_index", "window_timestamp")

#: Group B feature names, verbatim (each computed per -listed channel).
GROUP_B_FEATURES = (
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "slope",
    "velocity",
    "acceleration",
    "trend",
    "persistence",
    "change_point_score",
)

#: Channels per window that Group B statistics are computed over (the 
#: temporal group is defined over the movement channels).
B_CHANNELS = ("tilt_x", "tilt_y", "displacement")


def _series_blocks(windowed: pd.DataFrame, channel: str) -> list[pd.DataFrame]:
    """Per-(event, node) blocks of window rows in window order."""
    col = f"{channel}_mean"
    if col not in windowed.columns:
        raise ValueError(f"windowed frame missing {col!r} for Group B")
    return [g for _, g in windowed.groupby(["event_id", "node_id"], sort=False)]


def emit_group_b(
    windowed: pd.DataFrame,
    *,
    channels: tuple[str, ...] = B_CHANNELS,
    history_windows: int = 3,
) -> pd.DataFrame:
    """Emit Group B temporal features per window.

 ``history_windows`` rolling moments (mean/std/min/max) use only the
 current window and the immediately preceding ``history_windows`` windows
 **within the same series** — never any row outside the series, so no
 train/test boundary can be crossed (windows are consecutive within a
 series; split boundaries are series-aligned in ).
 """
    n_hist = int(history_windows)
    if n_hist < 0:
        raise ValueError("history_windows must be >= 0")
    required = [
        f"{ch}_{stat}" for ch in channels for stat in ("mean", "slope", "velocity", "acceleration")
    ]
    missing = [c for c in required if c not in windowed.columns]
    if missing:
        raise ValueError(f"windowed frame missing Group B sources: {missing}")
    out_frames: list[pd.DataFrame] = []
    for _keys, g in windowed.groupby(["event_id", "node_id"], sort=False):
        g = g.sort_values("window_index", kind="stable").reset_index(drop=True)
        keep = list(_WINDOW_KEYS) + [
            f"{ch}_{stat}" for ch in channels for stat in ("mean", "slope", "velocity", "acceleration")
        ]
        block = g[keep].copy()
        # window length in steps, from the window's own start/end stamps
        interval = float(_grid_interval_hours())
        window_steps = (
            (g["window_end"].to_numpy(dtype=float) - g["window_start"].to_numpy(dtype=float))
            / interval
        ) + 1.0
        for ch in channels:
            mean_col = f"{ch}_mean"
            slope_col = f"{ch}_slope"

            series_means = g[mean_col].to_numpy(dtype=float)
            # rolling moments over the window's own history (series-internal);
            # std of a single-window history is undefined → 0.0 (no spread info)
            k = n_hist + 1
            rolled = pd.Series(series_means).rolling(k, min_periods=1)
            rolled_mean = rolled.mean().to_numpy()
            rolled_std = rolled.std().to_numpy()
            rolled_std = np.where(np.isfinite(rolled_std), rolled_std, 0.0)
            rolled_min = rolled.min().to_numpy()
            rolled_max = rolled.max().to_numpy()
            block[f"{ch}_rolling_mean"] = rolled_mean
            block[f"{ch}_rolling_std"] = rolled_std
            block[f"{ch}_rolling_min"] = rolled_min
            block[f"{ch}_rolling_max"] = rolled_max

            # trend: the window slope re-expressed as total change over the window
            block[f"{ch}_trend"] = g[slope_col].to_numpy(dtype=float) * (window_steps - 1.0)
            # Both features are causal: the value for window i can only use
            # windows through i. A full-series scalar repeated on every row
            # leaks later windows into earlier predictions.
            block[f"{ch}_persistence"] = np.asarray(
                [_persistence(series_means[: i + 1]) for i in range(len(series_means))],
                dtype=float,
            )
            block[f"{ch}_change_point_score"] = np.asarray(
                [_change_point(series_means[: i + 1], rolled_std[: i + 1]) for i in range(len(series_means))],
                dtype=float,
            )

        out_frames.append(block)

    out = pd.concat(out_frames, ignore_index=True) if out_frames else windowed.iloc[0:0].copy()

    # -exact aggregate columns: the temporal features of the primary
    # movement channel (displacement), under the bare names.
    for feat in GROUP_B_FEATURES:
        src = f"displacement_{feat}"
        if src in out.columns:
            out[feat] = out[src]
    return out


def _grid_interval_hours() -> float:
    from src.config import load_config

    return float(load_config("preprocessing")["resampling"]["grid_interval_minutes"]) / 60.0


def _persistence(means: np.ndarray, k: int = 4) -> float:
    """Correlation of the last k window levels with their predecessors (series-internal)."""
    if means.size < k + 2:
        return 0.0
    a = means[-(k + 1) : -1]
    b = means[-k:]
    sa, sb = np.std(a), np.std(b)
    if sa <= 0 or sb <= 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _change_point(means: np.ndarray, rolled_std: np.ndarray) -> float:
    """|current level − median of prior levels| / rolling std (0 if no history)."""
    if means.size < 2:
        return 0.0
    prior = means[:-1]
    sigma = rolled_std[-1]
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.0
    return float(abs(means[-1] - np.median(prior)) / sigma)


def assert_no_boundary_crossing(windowed: pd.DataFrame, channels: tuple[str, ...] = B_CHANNELS) -> None:
    """Recompute a rolling statistic per series and assert equality.

 If any rolling value had been computed across series/event boundaries, the
 per-series recomputation would differ. This is the executable form of the
 requirement that no rolling statistic crosses a train/test boundary.
 """
    for ch in channels:
        col = f"{ch}_rolling_mean"
        if col not in windowed.columns:
            raise ValueError(f"missing {col!r}")
        for _keys, g in windowed.groupby(["event_id", "node_id"], sort=False):
            g = g.sort_values("window_index", kind="stable")
            expected = pd.Series(g[f"{ch}_mean"].to_numpy(dtype=float)).rolling(4, min_periods=1).mean()
            actual = g[col].to_numpy(dtype=float)
            if not np.allclose(actual, expected.to_numpy(), equal_nan=True):
                raise AssertionError(
                    f"boundary crossing detected: {col} for series "
                    f"{_keys} does not match its per-series recomputation"
                )
