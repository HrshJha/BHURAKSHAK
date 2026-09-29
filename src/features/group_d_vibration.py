"""Feature Group D — Vibration — Group D,.

 Group D: RMS, peak, crest_factor, low/mid/high band energy,
spectral_centroid.

 acceptance-critical constraint: on-node accelerometers sample at ≥100 Hz
but the node **summarises to 1 Hz before transmission** — the mesh only ever
carries ``vibration_rms`` and ``vibration_peak``. Features are therefore
computed from the summarised series **only**; the module refuses any input
that still carries raw high-rate samples (a ``raw_samples``/``raw_waveform``
column raises), and band energies are honestly derived from the 1 Hz-summarised
signal (the honest ceiling of what the mesh can know — no invented spectra).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["GROUP_D_FEATURES", "emit_group_d"]

GROUP_D_FEATURES = (
    "vibration_rms",
    "vibration_peak",
    "crest_factor",
    "band_energy_low",
    "band_energy_mid",
    "band_energy_high",
    "spectral_centroid",
)

_FORBIDDEN_RAW_COLUMNS = ("raw_samples", "raw_waveform", "raw_acceleration")

_WINDOW_KEYS = ("event_id", "node_id", "window_index", "window_timestamp")


def emit_group_d(windowed: pd.DataFrame) -> pd.DataFrame:
    """Emit Group D features per window from the summarised vibration series.

 ``windowed`` is the windowing engine's output: per-window
 ``vibration_rms_*``/``vibration_peak_*`` statistics over the 60-step
 window. Features:

 - ``vibration_rms`` / ``vibration_peak``: window level (mean of the
 per-step summarised channels — the names verbatim);
 - ``crest_factor``: peak / rms (both window means) — shock impulsive-ness;
 - ``band_energy_low/mid/high`` + ``spectral_centroid``: derived from the
 per-window RMS *trend* across steps (its within-window mean, slope and
 variability), the frequency content the 1 Hz summarised stream exposes.
 """
    for col in _FORBIDDEN_RAW_COLUMNS:
        if col in windowed.columns:
            raise ValueError(
                f"Group D must never see raw high-rate samples ({col!r}); "
                ": nodes summarise vibration to 1 Hz before transmission"
            )
    required = [f"vibration_{s}" for s in ("rms_mean", "rms_std", "rms_slope", "peak_mean", "peak_max")]
    missing = [c for c in required if c not in windowed.columns]
    if missing:
        raise ValueError(f"windowed frame missing Group D sources: {missing}")

    out = windowed[list(_WINDOW_KEYS)].copy()
    rms_mean = windowed["vibration_rms_mean"].to_numpy(dtype=float)
    rms_std = windowed["vibration_rms_std"].to_numpy(dtype=float)
    rms_slope = np.abs(windowed["vibration_rms_slope"].to_numpy(dtype=float))
    peak_mean = windowed["vibration_peak_mean"].to_numpy(dtype=float)
    peak_max = windowed["vibration_peak_max"].to_numpy(dtype=float)

    out["vibration_rms"] = rms_mean
    out["vibration_peak"] = peak_max
    out["crest_factor"] = np.where(rms_mean > 0, peak_mean / rms_mean, 0.0)

    # band energies from the summarised window trend: low = slow level (mean),
    # mid = within-window variability (std), high = impulsive change — the RMS
    # trend slope plus the excess of the window's PEAK PEAK over its TYPICAL
    # peak (a burst shows up as a spiky peak, not as the noise-floor crest).
    # Units are energy-like (squared) and normalised to sum to 1 per window.
    low = rms_mean**2
    mid = rms_std**2
    high = (rms_slope**2) + np.maximum(peak_max**2 - peak_mean**2, 0.0)
    total = low + mid + high
    safe = np.where(total > 0, total, 1.0)
    out["band_energy_low"] = low / safe
    out["band_energy_mid"] = mid / safe
    out["band_energy_high"] = high / safe

    # spectral centroid: energy-weighted mean band index (0=low, 1=mid, 2=high);
    # bands already sum to 1, so the centroid is just mid + 2·high (0 when quiet).
    out["spectral_centroid"] = out["band_energy_mid"] + 2.0 * out["band_energy_high"]
    return out
