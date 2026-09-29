""" acceptance tests — tilt is the gradient of W, not independently sampled."""

from __future__ import annotations

import numpy as np

from src.config import physics_config
from src.simulator.channels_tilt import finite_difference_gradient, generate_tilt, tilt_magnitude
from src.simulator.deformation_field import DeformationField, FieldParams


def _field() -> DeformationField:
    return DeformationField(FieldParams.from_config())


def test_tilt_tracks_analytic_gradient() -> None:
    """Pearson r between generated tilt and the analytic gradient > 0.99."""
    f = _field()
    rng = np.random.default_rng(42)
    n = 400
    xs = rng.uniform(-200, 200, n)
    ys = rng.uniform(-200, 200, n)
    t = 10.0

    tilt_x, tilt_y = generate_tilt(f, xs, ys, t, rng)
    gx, gy = finite_difference_gradient(f, xs, ys, t)

    r_x = np.corrcoef(tilt_x, np.degrees(gx / 1000.0))[0, 1]
    r_y = np.corrcoef(tilt_y, np.degrees(gy / 1000.0))[0, 1]
    assert r_x > 0.99, f"tilt_x must track ∂W/∂x (r={r_x:.4f})"
    assert r_y > 0.99, f"tilt_y must track ∂W/∂y (r={r_y:.4f})"


def test_tilt_noise_only_from_config() -> None:
    f = _field()
    rng = np.random.default_rng(1)
    xs = np.zeros(2000)
    ys = np.zeros(2000)
    t = 5.0
    tilt_x, _ = generate_tilt(f, xs, ys, t, rng)
    # at the panel center the gradient is 0, so the sample IS the noise
    std = float(np.std(tilt_x))
    expected = float(physics_config()["noise"]["tilt_noise_std_deg"])
    assert abs(std - expected) < 0.2 * expected, (
        f"noise std {std:.4f} must match config {expected:.4f}"
    )


def test_tilt_magnitude_is_hypotenuse() -> None:
    tx = np.array([3.0, 0.0])
    ty = np.array([4.0, 0.0])
    assert np.allclose(tilt_magnitude(tx, ty), [5.0, 0.0])


def test_tilt_zero_at_center_exactly() -> None:
    """At the subsidence centre the gradient vanishes: tilt = noise only."""
    f = _field()
    rng = np.random.default_rng(3)
    t = 8.0
    tilt_x, tilt_y = generate_tilt(f, np.array([0.0]), np.array([0.0]), t, rng)
    gx, gy = finite_difference_gradient(f, np.array([0.0]), np.array([0.0]), t)
    assert float(np.degrees(gx[0] / 1000.0)) == 0.0
    assert float(np.degrees(gy[0] / 1000.0)) == 0.0


def test_tilt_antisymmetric_across_center() -> None:
    """The bowl's gradient is antisymmetric: tilt at (−d, 0) and (+d, 0) oppose.
 (Sign convention is defined by tilt = degrees(∂W/∂x / 1000); the physics
 requirement is that the two sides disagree in sign, not which is which.)"""
    f = _field()
    rng = np.random.default_rng(5)
    t = 12.0
    left_x, _ = generate_tilt(f, np.array([-100.0]), np.array([0.0]), t, rng)
    right_x, _ = generate_tilt(f, np.array([100.0]), np.array([0.0]), t, rng)
    assert float(left_x[0]) * float(right_x[0]) < 0.0, "tilt must flip sign across the centre"
    # and the clean signal must dominate the configured noise
    expected_std = float(physics_config()["noise"]["tilt_noise_std_deg"])
    assert abs(float(left_x[0])) > 5 * expected_std
