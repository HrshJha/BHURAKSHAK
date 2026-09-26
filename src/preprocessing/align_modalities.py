"""Cross-modality temporal alignment — PRD §9.1 + FR-13 (T-031).

§9.1: "Sentinel-1 InSAR passes and DGPS observations are asynchronous and
low-frequency relative to the sensor mesh. They are aligned to the nearest
sensor feature window by timestamp (default tolerance: InSAR ±12 hours, DGPS
±1 hour) and tagged with their own observation timestamp so staleness is
visible rather than implied."

An alignment joins each external observation (InSAR/DGPS) to the nearest
feature-window centre; the join succeeds only if the offset is within the
modality's configured tolerance. Every aligned record keeps
``observation_timestamp`` (its own) and ``window_timestamp`` (the joined
window's) plus an explicit ``staleness_hours`` — staleness is visible, never
implied. Observations outside tolerance are reported, not force-joined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import load_config

__all__ = [
    "AlignmentError",
    "AlignmentResult",
    "align_to_windows",
    "tolerance_for",
]

#: Modality names with §9.1 default tolerances (hours) — resolved from config.
MODALITIES = ("insar", "dgps")


class AlignmentError(ValueError):
    """Raised on invalid alignment inputs or unknown modalities."""


@dataclass
class AlignmentResult:
    df: pd.DataFrame  # aligned records (joined rows only)
    n_observations: int
    n_aligned: int
    n_out_of_tolerance: int
    max_staleness_hours: float

    @property
    def n_unaligned(self) -> int:
        return self.n_observations - self.n_aligned


def tolerance_for(modality: str) -> float:
    """Configured §9.1 alignment tolerance in hours for a modality."""
    cfg = load_config("preprocessing")["align_modalities"]
    key = f"{modality}_tolerance_hours"
    if key not in cfg:
        raise AlignmentError(f"unknown modality {modality!r}; expected one of {list(MODALITIES)}")
    tol = float(cfg[key])
    if tol <= 0:
        raise AlignmentError(f"tolerance for {modality!r} must be > 0")
    return tol


def align_to_windows(
    observations: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    modality: str,
) -> AlignmentResult:
    """Align external observations to the nearest feature-window centre.

    ``observations`` must carry ``observation_timestamp`` (hours) plus any
    payload columns (e.g. ``LOS_displacement``). ``windows`` must carry
    ``window_timestamp`` (hours). Each observation joins the nearest window
    within the modality's tolerance; otherwise it is counted as out of
    tolerance and excluded (staleness is never hidden by force-joining).
    """
    tol = tolerance_for(modality)
    for col in ("observation_timestamp",):
        if col not in observations.columns:
            raise AlignmentError(f"observations must carry column {col!r}")
    if "window_timestamp" not in windows.columns:
        raise AlignmentError("windows must carry column 'window_timestamp'")

    obs_ts = pd.to_numeric(observations["observation_timestamp"], errors="coerce").to_numpy(dtype=float)
    win_ts = pd.to_numeric(windows["window_timestamp"], errors="coerce").to_numpy(dtype=float)
    win_ts = win_ts[~np.isnan(win_ts)]
    if win_ts.size == 0:
        return AlignmentResult(
            df=observations.iloc[0:0].copy(),
            n_observations=int(np.isnan(obs_ts).sum() + (~np.isnan(obs_ts)).sum()),
            n_aligned=0,
            n_out_of_tolerance=int((~np.isnan(obs_ts)).sum()),
            max_staleness_hours=float("nan"),
        )

    n = obs_ts.size
    aligned_rows: list[pd.DataFrame] = []
    n_aligned = 0
    n_out = 0
    max_stale = 0.0
    for i in range(n):
        t = obs_ts[i]
        if np.isnan(t):
            n_out += 1
            continue
        j = int(np.argmin(np.abs(win_ts - t)))
        delta = abs(float(win_ts[j]) - float(t))
        if delta > tol:
            n_out += 1
            continue
        row = observations.iloc[[i]].copy().reset_index(drop=True)
        row["window_timestamp"] = float(win_ts[j])
        row["staleness_hours"] = float(win_ts[j]) - float(t)  # signed: window minus observation
        aligned_rows.append(row)
        n_aligned += 1
        max_stale = max(max_stale, abs(delta))

    out = (
        pd.concat(aligned_rows, ignore_index=True)
        if aligned_rows
        else observations.iloc[0:0].copy()
    )
    return AlignmentResult(
        df=out,
        n_observations=n,
        n_aligned=n_aligned,
        n_out_of_tolerance=n_out,
        max_staleness_hours=max_stale if n_aligned else float("nan"),
    )
