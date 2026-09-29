""" acceptance tests — spatial kernel matches formula exactly."""

from __future__ import annotations

import math

import numpy as np

from src.config import physics_config
from src.simulator.spatial_model import (
    analytic_kernel_check,
    sigma_from_config,
    spatial_kernel,
)


def test_kernel_matches_closed_form_within_1e_9() -> None:
    cfg = physics_config()
    p = cfg["physics"]
    sigma = sigma_from_config(cfg)
    rng = np.random.default_rng(42)
    xs = rng.uniform(-300, 300, size=25)
    ys = rng.uniform(-300, 300, size=25)

    got = spatial_kernel(xs, ys, p["panel_center_x"], p["panel_center_y"], sigma)
    for xv, yv, gv in zip(xs, ys, got):
        expected = math.exp(
            -((xv - p["panel_center_x"]) ** 2 + (yv - p["panel_center_y"]) ** 2)
            / (2.0 * sigma**2)
        )
        assert abs(gv - expected) < 1e-9


def test_kernel_peaks_at_center() -> None:
    cfg = physics_config()
    p = cfg["physics"]
    sigma = sigma_from_config(cfg)
    center = spatial_kernel(p["panel_center_x"], p["panel_center_y"], p["panel_center_x"], p["panel_center_y"], sigma)
    off = spatial_kernel(p["panel_center_x"] + 7.0, p["panel_center_y"] - 3.0, p["panel_center_x"], p["panel_center_y"], sigma)
    assert center == pytest_approx(1.0)
    assert off < center


def pytest_approx(value: float) -> float:
    return value


def test_kernel_monotonically_decays_with_distance() -> None:
    sigma = sigma_from_config()
    dists = np.linspace(0, 200, 21)
    values = spatial_kernel(dists, np.zeros_like(dists), 0.0, 0.0, sigma)
    assert np.all(np.diff(values) <= 0), "kernel must be non-increasing with distance"


def test_analytic_self_check_20_points() -> None:
    assert analytic_kernel_check(0.0, 0.0, sigma_from_config(), n_points=20)


def test_sigma_is_config_derived_not_hardcoded() -> None:
    cfg = physics_config()
    expected = float(cfg["physics"]["influence_radius"]) / float(cfg["kernel"]["sigma_divisor"])
    assert sigma_from_config(cfg) == expected


def test_invalid_sigma_raises() -> None:
    try:
        spatial_kernel(1.0, 1.0, 0.0, 0.0, 0.0)
    except ValueError:
        return
    raise AssertionError("sigma=0 must raise ValueError")
