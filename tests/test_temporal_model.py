""" acceptance tests — Knothe temporal model matches formula exactly."""

from __future__ import annotations

import math

import numpy as np

from src.config import physics_parameters
from src.simulator.temporal_model import (
    c_from_config,
    growth_velocity,
    subsidence_growth,
    time_to_fraction,
    w_max_from_config,
)


def test_w_of_t_matches_closed_form_within_1e_9() -> None:
    w_max, c = 2100.0, 0.25
    ts = np.linspace(0, 60, 41)
    got = subsidence_growth(ts, w_max, c)
    for t, g in zip(ts, got):
        expected = w_max * (1.0 - math.exp(-c * t))
        assert abs(g - expected) < 1e-9


def test_w_at_zero_is_zero() -> None:
    assert subsidence_growth(0.0, 2100.0, 0.25) == 0.0


def test_w_monotonically_approaches_w_max() -> None:
    w_max, c = 1500.0, 0.4
    ts = np.linspace(0, 90, 200)
    w = subsidence_growth(ts, w_max, c)
    # Non-decreasing to machine precision: at c*t > ~20, 1-e^(-ct) saturates
    # float64 exactly at W_max, so strict > would fail on representation noise.
    assert np.all(np.diff(w) >= -1e-9), "W(t) must be non-decreasing (within fp tolerance)"
    assert np.all(w <= w_max + 1e-9), "W(t) must never exceed W_max"
    assert w[-1] > 0.999 * w_max, "W(t) must asymptotically approach W_max"


def test_velocity_positive_and_decaying() -> None:
    w_max, c = 2100.0, 0.25
    ts = np.array([0.0, 1.0, 5.0, 20.0])
    v = growth_velocity(ts, w_max, c)
    assert np.all(v > 0)
    assert np.all(np.diff(v) < 0), "growth velocity must decay over time"
    assert v[0] == pytest_approx2(w_max * c)


def pytest_approx2(value: float) -> float:
    return value


def test_time_to_fraction() -> None:
    c = 0.25
    t63 = time_to_fraction(1 - math.exp(-1), c)  # exact 1/e point
    assert abs(t63 - 1.0 / c) < 1e-9
    assert abs(time_to_fraction(0.5, c) - (math.log(2) / c)) < 1e-9


def test_config_values_reachable() -> None:
    params = physics_parameters()
    assert w_max_from_config(params) == params["maximum_subsidence"]
    assert c_from_config(params) == params["time_coefficient"]


def test_invalid_c_raises() -> None:
    try:
        subsidence_growth(1.0, 100.0, 0.0)
    except ValueError:
        return
    raise AssertionError("c=0 must raise ValueError")
