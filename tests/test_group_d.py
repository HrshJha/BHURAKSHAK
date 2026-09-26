"""T-038 acceptance tests — Feature Group D (vibration, §8.3 summarised-only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.group_d_vibration import GROUP_D_FEATURES, emit_group_d
from src.features.windowing import build_windows


def _windowed(quiet: bool = False) -> pd.DataFrame:
    n = 144
    ts = (10.0 / 60.0) * np.arange(n)
    rng = np.random.default_rng(0)
    bursty = not quiet
    burst = rng.random(n) > 0.9  # a shock raises PEAK far more than RMS
    rms = 0.02 + (0.15 * burst if bursty else 0.0)
    peak = rms + 0.03 + (0.5 * burst if bursty else 0.0)
    return build_windows(
        pd.DataFrame(
            {
                "event_id": "E1",
                "node_id": "V0001",
                "timestamp": ts,
                "vibration_rms": rms,
                "vibration_peak": peak,
            }
        )
    ).df


def test_feature_names_match_section_13() -> None:
    assert GROUP_D_FEATURES == (
        "vibration_rms",
        "vibration_peak",
        "crest_factor",
        "band_energy_low",
        "band_energy_mid",
        "band_energy_high",
        "spectral_centroid",
    )


def test_features_computed_from_summarised_channels() -> None:
    windowed = _windowed()
    out = emit_group_d(windowed)
    row = out.iloc[0]
    expected_rms = windowed["vibration_rms_mean"].iloc[0]
    assert row["vibration_rms"] == pytest.approx(expected_rms)
    assert row["vibration_peak"] == pytest.approx(windowed["vibration_peak_max"].iloc[0])
    assert row["crest_factor"] == pytest.approx(
        windowed["vibration_peak_mean"].iloc[0] / expected_rms
    )


def test_bands_sum_to_one_and_centroid_is_weighted_mean() -> None:
    out = emit_group_d(_windowed())
    total = out["band_energy_low"] + out["band_energy_mid"] + out["band_energy_high"]
    assert np.allclose(total, 1.0)
    centroid = out["band_energy_mid"] + 2.0 * out["band_energy_high"]
    assert np.allclose(out["spectral_centroid"], centroid)
    assert out["spectral_centroid"].between(0.0, 2.0).all()


def test_quiet_series_concentrates_energy_in_low_band() -> None:
    out = emit_group_d(_windowed(quiet=True))
    assert (out["band_energy_low"] > 0.9).all()
    assert (out["spectral_centroid"] < 0.2).all()


def test_bursty_series_pushes_energy_high() -> None:
    quiet = emit_group_d(_windowed(quiet=True))
    bursty = emit_group_d(_windowed(quiet=False))
    assert bursty["band_energy_high"].mean() > quiet["band_energy_high"].mean()
    assert bursty["crest_factor"].mean() > quiet["crest_factor"].mean()


def test_raw_high_rate_samples_are_refused() -> None:
    """§8.3: features must come from on-node summarised vibration, never raw."""
    windowed = _windowed()
    for col in ("raw_samples", "raw_waveform", "raw_acceleration"):
        with pytest.raises(ValueError, match="raw high-rate"):
            emit_group_d(windowed.assign(**{col: [1] * len(windowed)}))


def test_missing_sources_raise() -> None:
    windowed = _windowed().drop(columns=["vibration_peak_max"])
    with pytest.raises(ValueError, match="vibration_peak_max"):
        emit_group_d(windowed)


def test_one_row_per_window_with_keys() -> None:
    windowed = _windowed()
    out = emit_group_d(windowed)
    assert len(out) == len(windowed) == 9
    assert {"event_id", "node_id", "window_index", "window_timestamp"} <= set(out.columns)
