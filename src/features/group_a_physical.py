"""Feature Group A — Physical — Group A,.

 Group A, exactly: ``tilt_x, tilt_y, tilt_magnitude, displacement, strain``.

Mapping from the windowing engine (recorded deliberately): each name
is the **window mean** of the corresponding raw channel over the 60-step
model window. The temporal dynamics of the same channels are carried by
Group B (slope/velocity/acceleration), so Group A is the window's physical
state. The mapping is one-to-one and naming-exact — the acceptance check
asserts the emitted names match verbatim.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["GROUP_A_FEATURES", "GROUP_A_SOURCES", "emit_group_a"]

#: Group A feature names, verbatim.
GROUP_A_FEATURES = ("tilt_x", "tilt_y", "tilt_magnitude", "displacement", "strain")

#: name → windowed-frame source column (window mean of the raw channel).
GROUP_A_SOURCES = {
    "tilt_x": "tilt_x_mean",
    "tilt_y": "tilt_y_mean",
    "tilt_magnitude": "tilt_magnitude_mean",
    "displacement": "displacement_mean",
    "strain": "strain_mean",
}

_WINDOW_KEYS = ("event_id", "node_id", "window_index", "window_timestamp")


def emit_group_a(windowed: pd.DataFrame) -> pd.DataFrame:
    """Emit Group A features for a windowed frame (one row per window).

 The output keeps the window keys and carries exactly the five -named
 columns. Raises if the windowed frame lacks a source channel — fail
 loudly, never emit silent NaN features.
 """
    missing = [src for src in GROUP_A_SOURCES.values() if src not in windowed.columns]
    if missing:
        raise ValueError(f"windowed frame is missing Group A sources: {missing}")
    out = windowed[list(_WINDOW_KEYS)].copy()
    for name, src in GROUP_A_SOURCES.items():
        out[name] = windowed[src]
    return out
