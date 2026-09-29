""" acceptance tests — five injectable fault modes, each tagged SENSOR_FAULT."""

from __future__ import annotations

import numpy as np

from src.simulator.faults import FAULT_LABEL, FaultParams, FaultType, all_fault_types, inject_fault


def _series(n: int = 60) -> np.ndarray:
    return np.linspace(0.0, 10.0, n)  # gently rising "ground movement"


def _params() -> FaultParams:
    return FaultParams(
        bias_mm=2.0,
        drift_mm_per_day=1.5,
        dropout_probability=0.9,
        dropout_run_length=5,
        spike_probability=0.3,
        spike_magnitude_mm=25.0,
        stuck_duration_steps=10,
    )


def test_five_fault_modes_exist() -> None:
    types = all_fault_types()
    assert {t.name for t in types} == {"BIAS", "STUCK", "DROPOUT", "SPIKE", "DRIFT"}


def test_bias_adds_constant_offset() -> None:
    v0 = _series()
    rng = np.random.default_rng(0)
    r = inject_fault(v0, FaultType.BIAS, rng, onset_step=30, params=_params())
    assert np.allclose(r.values[:30], v0[:30]), "pre-onset untouched"
    assert np.allclose(r.values[30:] - v0[30:], 2.0), "BIAS adds the constant configured offset"
    assert r.fault_mask[30:].all() and not r.fault_mask[:30].any()
    assert r.fault_type.value in {"BIAS"} and FAULT_LABEL == "SENSOR_FAULT"


def test_stuck_flatlines_at_pre_fault_level() -> None:
    v0 = _series()
    rng = np.random.default_rng(0)
    r = inject_fault(v0, FaultType.STUCK, rng, onset_step=20, params=_params())
    window = r.values[20:30]
    assert np.allclose(window, window[0]), "stuck readings must be flat"
    assert np.allclose(r.values[30:], v0[30:]), "sensor recovers after stuck window"


def test_dropout_produces_nan_runs() -> None:
    v0 = _series()
    rng = np.random.default_rng(1)
    r = inject_fault(v0, FaultType.DROPOUT, rng, onset_step=0, params=_params())
    assert np.isnan(r.values).sum() > 0, "dropout must produce NaN runs"
    assert np.isnan(r.values).sum() < r.values.size, "dropout need not swallow everything"


def test_spike_produces_isolated_extremes() -> None:
    v0 = _series()
    rng = np.random.default_rng(2)
    r = inject_fault(v0, FaultType.SPIKE, rng, onset_step=0, params=_params())
    dev = np.abs(np.nan_to_num(r.values) - v0)
    spiked = dev > 10.0
    assert spiked.any(), "spikes must be present"
    assert not spiked.all(), "spikes must be isolated, not sustained"


def test_drift_grows_linearly() -> None:
    v0 = _series()
    rng = np.random.default_rng(3)
    r = inject_fault(v0, FaultType.DRIFT, rng, onset_step=10, days_per_step=1.0, params=_params())
    offsets = r.values[10:] - v0[10:]
    assert np.all(np.diff(offsets) > 0), "drift must grow over time"
    assert np.isclose(offsets[0], 0.0), "no jump at onset — drift starts at zero"


def test_every_mode_masks_only_affected_samples() -> None:
    for ft in all_fault_types():
        rng = np.random.default_rng(4)
        r = inject_fault(_series(), ft, rng, onset_step=30, params=_params())
        assert r.fault_mask.dtype == bool
        assert not r.fault_mask[:30].any(), f"{ft}: pre-onset samples never fault-tagged"


def test_invalid_onset_raises() -> None:
    rng = np.random.default_rng(0)
    try:
        inject_fault(_series(), FaultType.BIAS, rng, onset_step=999, params=_params())
    except ValueError:
        return
    raise AssertionError("out-of-range onset_step must raise")
