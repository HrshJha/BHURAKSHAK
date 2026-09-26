"""Sensor-health state detection — PRD §9 + §8.4 (T-033).

§9: the pipeline must tolerate "sensor drift, stuck sensors, sudden sensor
offsets — treated as first-class data-quality states ... because in the
synthetic-data phase (§10) they are deliberately injected as ``SENSOR_FAULT``
scenarios and must be distinguishable from real ground movement".

§8.4 item 6: sensors whose recalibration offset exceeds a configurable
threshold are flagged ``sensor_health = DEGRADED`` rather than silently
trusted.

States emitted per (event, node, channel) series:

- ``HEALTHY``   — no fault condition fires;
- ``STUCK``     — a run of ≥ ``stuck_min_repeats`` near-identical readings
                  (|Δ| ≤ ``stuck_value_tolerance``);
- ``OFFSET``    — a sudden level shift larger than ``offset_jump_sigma`` ×
                  global per-sample noise σ (step-shaped);
- ``DRIFT``     — a statistically significant TREND (OLS |t| above
                  ``drift_score_threshold``) that is ramp-shaped rather than
                  step-shaped;
- ``DEGRADED``  — a mild offset (above ``degraded_offset_jump_sigma`` σ but
                  below the hard threshold) or an emerging trend — flagged,
                  not silently trusted (§8.4 item 6).

Two statistics drive the classification, both normalised by a robust
per-sample noise σ estimated from first differences (immune to level, trend
and jumps): the trend t-statistic and the maximum window-median level shift.
A shape discriminator separates ramps from steps: a pure ramp has trend ≫ any
step it could produce (|t| / jump ≈ n/5.8), a step has |t| ≈ 5.2 × jump.

Robust statistics (median / MAD) are used throughout so genuine gradual ground
movement does not by itself trip the hard fault states; the §10 ``fault_label``
remains ground truth, and this table is the preprocessing-time separation of
``SENSOR_FAULT`` from real movement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import load_config

__all__ = [
    "SensorHealthError",
    "SeriesHealth",
    "HEALTHY",
    "STUCK",
    "DRIFT",
    "OFFSET",
    "DEGRADED",
    "sensor_health_params",
    "detect_series_health",
    "flag_series_health",
    "longest_stuck_run",
    "trend_t_statistic",
]

HEALTHY = "HEALTHY"
STUCK = "STUCK"
DRIFT = "DRIFT"
OFFSET = "OFFSET"
DEGRADED = "DEGRADED"


class SensorHealthError(ValueError):
    """Raised on invalid sensor-health inputs or configuration."""


@dataclass(frozen=True)
class SeriesHealth:
    """Detection result for one (event, node, channel) series."""

    state: str
    stuck_run_len: int
    drift_score: float
    offset_jump_sigma: float
    detail: str


def sensor_health_params() -> dict[str, float]:
    """The §8.4/§9 sensor-health thresholds from configs/preprocessing.yaml."""
    cfg = load_config("preprocessing")["sensor_health"]
    required = (
        "drift_score_threshold",
        "stuck_value_tolerance",
        "stuck_min_repeats",
        "offset_jump_sigma",
        "offset_window_steps",
    )
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SensorHealthError(f"sensor_health config missing keys: {missing}")
    if float(cfg["drift_score_threshold"]) <= 0 or float(cfg["offset_jump_sigma"]) <= 0:
        raise SensorHealthError("drift_score_threshold and offset_jump_sigma must be > 0")
    if int(cfg["stuck_min_repeats"]) < 2:
        raise SensorHealthError("stuck_min_repeats must be >= 2")
    return dict(cfg)


def _robust_sigma(values: np.ndarray) -> float:
    """MAD-based σ estimate (1.4826 × median absolute deviation)."""
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return 0.0
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    return 1.4826 * float(mad)


def longest_stuck_run(values: np.ndarray, tol: float) -> int:
    """Longest run of consecutive near-identical readings (public for Group E)."""
    if values.size == 0:
        return 0
    best = run = 1
    for i in range(1, values.size):
        if np.isfinite(values[i]) and np.isfinite(values[i - 1]) and abs(values[i] - values[i - 1]) <= tol:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return int(best)


def _max_level_shift(values: np.ndarray, window: int, sigma_per_sample: float) -> float:
    """Largest pre/post window-median level shift, in units of the GLOBAL
    per-sample noise σ.

    The σ is global (from first differences), not per-window: a sliding local σ
    would chase noise dips and inflate the statistic. A sudden jump of size J
    scores ≈ J/σ; a gradual ramp of total trend T scores only ≈ T·window/(n·σ).
    """
    n = values.size
    if window <= 0 or n < 2 * window or sigma_per_sample <= 0:
        return 0.0
    best = 0.0
    for k in range(window, n - window + 1):
        before = values[k - window : k]
        after = values[k : k + window]
        shift = abs(float(np.nanmedian(after)) - float(np.nanmedian(before)))
        best = max(best, shift)
    return best / sigma_per_sample


def _longest_stuck_run(values: np.ndarray, tol: float) -> int:  # private alias
    return longest_stuck_run(values, tol)


def trend_t_statistic(values: np.ndarray, sigma_per_sample: float) -> float:
    """OLS slope significance: |t| = |trend| / SE(trend).

    ``trend`` is the fitted change over the whole series; SE(trend) =
    σ_per_sample·√(12/n). Under pure noise t ~ N(0, 1) — so a threshold of 3.5
    is a <0.05% false-flag rate — while genuine trends score ≫ 3.5.
    """
    finite = np.where(np.isfinite(values), values, np.nan)
    n = finite.size
    ok = np.isfinite(finite)
    if ok.sum() < 3 or n < 2 or sigma_per_sample <= 0:
        return 0.0
    t = np.arange(n, dtype=float)[ok]
    v = finite[ok]
    t_mean = t.mean()
    denom = float(((t - t_mean) ** 2).sum())
    if denom <= 0:
        return 0.0
    slope = float(((t - t_mean) * (v - v.mean())).sum() / denom)
    trend = slope * (n - 1)
    se_trend = sigma_per_sample * np.sqrt(12.0 / n)
    return abs(trend) / se_trend


def detect_series_health(values: np.ndarray, params: dict[str, float] | None = None) -> SeriesHealth:
    """Classify one (event, node, channel) series into its §8.4/§9 health state."""
    p = params or sensor_health_params()
    stuck_tol = float(p["stuck_value_tolerance"])
    stuck_min = int(p["stuck_min_repeats"])
    drift_thr = float(p["drift_score_threshold"])
    jump_sigma = float(p["offset_jump_sigma"])
    win = int(p["offset_window_steps"])

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return SeriesHealth(HEALTHY, 0, 0.0, 0.0, "empty series")

    # Noise scale from first differences (robust to level, trend and jumps).
    stuck_run = _longest_stuck_run(arr, stuck_tol)
    sigma_diff = _robust_sigma(np.diff(arr))
    sigma_per_sample = sigma_diff / np.sqrt(2.0)
    if sigma_diff <= 0:
        sigma_diff = 1.0  # constant series: STUCK has already fired
        sigma_per_sample = 1.0
    # Two statistics:
    #   drift_score — trend significance |t| (noise ~ N(0,1));
    #   offset_jump — window-median level shift in σ units (jump J → ≈ J/σ).
    # A pure ramp and a sudden jump BOTH inflate the t-statistic, so the shape
    # discriminator below decides which state a significant series belongs to.
    drift_score = trend_t_statistic(arr, sigma_per_sample)
    offset_jump = _max_level_shift(arr, win, sigma_per_sample)
    quiet = float(p["quiet_jump_sigma"])
    ramp_ratio = float(p["ramp_ratio_for_drift"])
    degraded_jump = float(p["degraded_offset_jump_sigma"])
    degraded_drift = float(p["degraded_drift_score"])

    # Ramp-like ⇔ no meaningful shift at all, or trend far larger than any
    # step could produce (a step's t ≈ 5.2×jump; a ramp's t ≈ 20×jump here).
    ramp_like = offset_jump < quiet or (drift_score / max(offset_jump, 1e-9)) > ramp_ratio

    if stuck_run >= stuck_min:
        state, detail = STUCK, f"run of {stuck_run} identical readings"
    elif offset_jump > jump_sigma and not ramp_like:
        state, detail = OFFSET, f"level shift = {offset_jump:.1f}σ (> {jump_sigma:g}σ, step-like)"
    elif drift_score > drift_thr and ramp_like:
        state, detail = DRIFT, f"trend significance {drift_score:.1f} > {drift_thr:g}"
    elif offset_jump > degraded_jump or drift_score > degraded_drift:
        state, detail = DEGRADED, f"mild drift ({drift_score:.1f}) or mild offset ({offset_jump:.1f}σ)"
    else:
        state, detail = HEALTHY, "no fault condition fired"
    return SeriesHealth(state, stuck_run, drift_score, offset_jump, detail)


def flag_series_health(
    df: pd.DataFrame,
    *,
    channels: tuple[str, ...] = ("tilt_x", "tilt_y", "displacement"),
    params: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Attach per-series health states to a raw node table.

    Returns one row per (event_id, node_id, channel) with the detected
    ``health_state`` and the underlying scores. The detector never rewrites
    labels — §10's ``fault_label`` stays ground truth; this table is the
    preprocessing-time separation of ``SENSOR_FAULT`` from movement.
    """
    p = params or sensor_health_params()
    rows: list[dict[str, object]] = []
    for (event, node), g in df.groupby(["event_id", "node_id"], sort=False):
        g = g.sort_values("timestamp", kind="stable")
        for ch in channels:
            if ch not in df.columns:
                continue
            values = pd.to_numeric(g[ch], errors="coerce").to_numpy(dtype=float)
            health = detect_series_health(values, p)
            rows.append(
                {
                    "event_id": event,
                    "node_id": node,
                    "channel": ch,
                    "health_state": health.state,
                    "stuck_run_len": health.stuck_run_len,
                    "drift_score": health.drift_score,
                    "offset_jump_sigma": health.offset_jump_sigma,
                    "detail": health.detail,
                }
            )
    return pd.DataFrame(rows)
