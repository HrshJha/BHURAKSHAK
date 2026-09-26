"""T-030 acceptance tests — node clock-drift estimation and correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.clock_drift import (
    FLAG_INSUFFICIENT,
    FLAG_OK,
    FLAG_SUSPECT,
    ClockDriftError,
    apply_clock_correction,
    drift_log,
    estimate_clock_drift,
)

INTERVAL_H = 10.0 / 60.0


def _receipts(node: str, n: int, true_offset_s: float, jitter_s: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    receipt = 3600.0 * INTERVAL_H * np.arange(n, dtype=float)  # gateway seconds
    node_ts = receipt - true_offset_s + rng.normal(0.0, jitter_s, size=n)
    return pd.DataFrame(
        {"node_id": node, "node_timestamp": node_ts, "receipt_timestamp": receipt}
    )


def test_zero_drift_estimate_is_near_zero_and_corrected() -> None:
    est = estimate_clock_drift(_receipts("V1", 50, 0.0))[0]
    assert est.node_id == "V1"
    assert est.offset_seconds == pytest.approx(0.0, abs=1e-6)
    assert est.corrected is True
    assert est.flag == FLAG_OK


def test_constant_offset_is_recovered_by_median() -> None:
    est = estimate_clock_drift(_receipts("V2", 50, 12.0))[0]
    assert est.offset_seconds == pytest.approx(12.0, abs=0.5)


def test_drift_rate_detects_growing_offset() -> None:
    # node runs 4 s/hour slow relative to the gateway
    n = 100
    receipt = 3600.0 * INTERVAL_H * np.arange(n, dtype=float)
    drift_rate = 4.0  # seconds per hour
    node_ts = receipt - drift_rate * (receipt / 3600.0)
    df = pd.DataFrame({"node_id": "V3", "node_timestamp": node_ts, "receipt_timestamp": receipt})
    est = estimate_clock_drift(df)[0]
    assert est.drift_rate_seconds_per_hour == pytest.approx(drift_rate, rel=0.05)


def test_insufficient_samples_not_corrected() -> None:
    est = estimate_clock_drift(_receipts("V4", 5, 12.0))[0]
    assert est.corrected is False
    assert est.flag == FLAG_INSUFFICIENT
    assert np.isnan(est.offset_seconds)


def test_offset_beyond_max_flagged_not_applied() -> None:
    # config max_abs_offset_seconds = 30 → a 1-hour offset is suspect
    est = estimate_clock_drift(_receipts("V5", 50, 3600.0))[0]
    assert est.corrected is False
    assert est.flag == FLAG_SUSPECT


def test_per_node_estimates_are_independent() -> None:
    df = pd.concat([_receipts("V1", 50, 0.0), _receipts("V2", 50, 20.0)], ignore_index=True)
    ests = estimate_clock_drift(df)
    by_node = {e.node_id: e for e in ests}
    assert by_node["V1"].offset_seconds == pytest.approx(0.0, abs=0.5)
    assert by_node["V2"].offset_seconds == pytest.approx(20.0, abs=0.5)


def test_drift_log_has_one_row_per_node_and_is_exposed() -> None:
    df = pd.concat([_receipts("V1", 50, 0.0), _receipts("V2", 50, 20.0)], ignore_index=True)
    log = drift_log(estimate_clock_drift(df))
    assert list(log["node_id"]) == ["V1", "V2"]
    assert {"node_id", "offset_seconds", "corrected", "flag"} <= set(log.columns)


def test_correction_shifts_node_timestamps_onto_gateway_time() -> None:
    est = estimate_clock_drift(_receipts("V1", 50, 24.0))[0]
    hours = np.arange(6, dtype=float) * INTERVAL_H
    df = pd.DataFrame({"node_id": "V1", "timestamp": hours})
    out, applied = apply_clock_correction(df, [est])
    assert applied == ["V1"]
    expected_shift_s = -24.0  # offset subtracted
    assert np.allclose(
        out["timestamp"].to_numpy(), hours + expected_shift_s / 3600.0, atol=1e-9
    )


def test_uncorrected_nodes_are_left_untouched_and_reported() -> None:
    suspect = estimate_clock_drift(_receipts("Vbad", 50, 3600.0))[0]
    thin = estimate_clock_drift(_receipts("Vthin", 5, 10.0))[0]
    df = pd.DataFrame({"node_id": ["Vbad", "Vthin"], "timestamp": [1.0, 2.0]})
    out, applied = apply_clock_correction(df, [suspect, thin])
    assert applied == [], "no node had a usable estimate"
    assert np.allclose(out["timestamp"].to_numpy(), df["timestamp"].to_numpy())


def test_input_frame_is_never_mutated() -> None:
    est = estimate_clock_drift(_receipts("V1", 50, 24.0))[0]
    df = pd.DataFrame({"node_id": "V1", "timestamp": [1.0, 2.0]})
    before = df["timestamp"].to_numpy().copy()
    apply_clock_correction(df, [est])
    assert np.allclose(df["timestamp"].to_numpy(), before)


def test_missing_columns_raise() -> None:
    with pytest.raises(ClockDriftError):
        estimate_clock_drift(pd.DataFrame({"node_id": ["V1"], "node_timestamp": [0.0]}))


def test_missing_timestamp_column_raises() -> None:
    est = estimate_clock_drift(_receipts("V1", 50, 0.0))[0]
    with pytest.raises(ClockDriftError):
        apply_clock_correction(pd.DataFrame({"node_id": ["V1"], "t": [1.0]}), [est])


def test_correction_config_is_the_threshold_authority() -> None:
    from src.config import load_config

    cfg = load_config("preprocessing")["clock_drift"]
    assert cfg["min_samples_for_estimate"] == 20
    assert cfg["max_abs_offset_seconds"] == 30.0
