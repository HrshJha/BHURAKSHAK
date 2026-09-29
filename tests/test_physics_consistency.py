""" acceptance tests — physics engine (physics_residual, )."""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.consistency import PhysicsConsistency, physics_engine
from src.simulator.deformation_field import DeformationField


def test_engine_reuses_the_same_deformation_field() -> None:
    engine = physics_engine()
    field = DeformationField()
    assert engine.field.params.panel_center_x == field.params.panel_center_x
    assert engine.field.w_max == field.w_max, "expected deformation must reuse the  model"
    assert engine.field.params.time_coefficient == field.params.time_coefficient


def test_residual_is_zero_when_observation_equals_the_model() -> None:
    engine = physics_engine()
    x, y, t = 0.0, 0.0, 12.0
    expected = float(engine.expected_displacement_mm(x, y, t))
    residual = engine.displacement_residual(expected, x, y, t)
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_expected_displacement_matches_the_field_directly() -> None:
    engine = physics_engine()
    field = DeformationField()
    x = np.array([-50.0, 0.0, 60.0])
    y = np.array([10.0, -20.0, 30.0])
    t_h = np.array([6.0, 18.0, 30.0])
    np.testing.assert_allclose(
        engine.expected_displacement_mm(x, y, t_h),
        np.asarray(field.expected_displacement(x, y, t_h / 24.0)),
        rtol=1e-12,
    )


def test_hours_to_days_conversion_is_applied_once() -> None:
    """A common unit bug: applying the /24 conversion twice would give a
 near-zero W(t); the engine must convert exactly once."""
    engine = physics_engine()
    field = DeformationField()
    t_h = 24.0  # exactly one Knothe day
    expected_day1 = float(field.expected_displacement(0.0, 0.0, 1.0))
    assert float(engine.expected_displacement_mm(0.0, 0.0, t_h)) == pytest.approx(expected_day1)
    # Knothe: W(1 day) = W_max·(1 − e^(−c)) with c = 0.25/day ≈ 0.22·W_max
    assert expected_day1 == pytest.approx(field.w_max * (1.0 - float(np.exp(-field.params.time_coefficient))))


def test_residual_grows_with_observation_error() -> None:
    engine = physics_engine()
    x, y, t = 30.0, -40.0, 10.0
    truth = float(engine.expected_displacement_mm(x, y, t))
    r0 = engine.displacement_residual(truth, x, y, t)
    r1 = engine.displacement_residual(truth + 12.0, x, y, t)
    assert r0 == pytest.approx(0.0, abs=1e-9)
    assert r1 == pytest.approx(12.0, abs=1e-9)


def test_tilt_residual_zero_for_gradient_consistent_observation() -> None:
    engine = physics_engine()
    x, y, t = np.array([20.0, -30.0]), np.array([5.0, 15.0]), 15.0
    etx, ety = engine.expected_tilt(x, y, t)
    rtx, rty = engine.tilt_residual(etx, ety, x, y, t)
    np.testing.assert_allclose(rtx, 0.0, atol=1e-12)
    np.testing.assert_allclose(rty, 0.0, atol=1e-12)


def test_expected_tilt_is_the_analytic_gradient() -> None:
    """ coupling: expected tilt must equal the field's analytic gradient."""
    engine = physics_engine()
    field = DeformationField()
    x, y, t = 45.0, -25.0, 20.0
    etx, ety = engine.expected_tilt(x, y, t)
    assert float(etx) == pytest.approx(float(field.dW_dx(x, y, t / 24.0)))
    assert float(ety) == pytest.approx(float(field.dW_dy(x, y, t / 24.0)))


def test_field_values_are_config_driven() -> None:
    from src.config import physics_parameters

    engine = physics_engine()
    p = physics_parameters()
    assert engine.field.params.mine_depth == p["mine_depth"]
    assert engine.field.params.time_coefficient == p["time_coefficient"]
