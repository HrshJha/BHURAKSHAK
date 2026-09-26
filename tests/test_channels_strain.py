"""T-014 acceptance tests — strain from inter-node distance change under W."""

from __future__ import annotations

import numpy as np

from src.simulator.channels_strain import (
    displaced_positions,
    edge_strain,
    pairwise_distance_matrix,
    strain_matrix,
)
from src.simulator.deformation_field import DeformationField, FieldParams


def _field() -> DeformationField:
    return DeformationField(FieldParams.from_config())


def _grid():
    xs, ys = np.meshgrid(np.linspace(-100, 100, 6), np.linspace(-150, 150, 6))
    return xs.ravel(), ys.ravel()


def test_strain_zero_when_w_max_zero() -> None:
    """At t=0 there is no subsidence, so d'_ij == d_ij and strain == 0 everywhere."""
    f = _field()
    x, y = _grid()
    s = strain_matrix(f, x, y, t=0.0)
    off_diag = s[~np.eye(s.shape[0], dtype=bool)]
    assert np.allclose(off_diag, 0.0, atol=1e-12), "W=0 ⇒ strain must be exactly 0"


def test_strain_grows_with_time_and_symmetric() -> None:
    f = _field()
    x, y = _grid()
    s_early = strain_matrix(f, x, y, t=5.0)
    s_late = strain_matrix(f, x, y, t=40.0)
    assert np.nanmax(np.abs(s_late)) > np.nanmax(np.abs(s_early)) > 0.0
    assert np.allclose(s_early, s_early.T, equal_nan=True), "strain matrix must be symmetric"


def test_strain_is_fractional_distance_change() -> None:
    f = _field()
    t = 20.0
    x = np.array([-50.0, 50.0])
    y = np.array([0.0, 0.0])
    s = strain_matrix(f, x, y, t)
    d0 = pairwise_distance_matrix(x, y)[0, 1]
    x1, y1 = displaced_positions(f, x, y, t)
    d1 = np.hypot(x1[1] - x1[0], y1[1] - y1[0])
    assert np.isclose(s[0, 1], (d1 - d0) / d0, atol=1e-12)


def test_edge_strain_matches_matrix() -> None:
    f = _field()
    t = 15.0
    e = edge_strain(f, -60.0, 20.0, 40.0, -30.0, t)
    s = strain_matrix(f, np.array([-60.0, 40.0]), np.array([20.0, -30.0]), t)
    assert np.isclose(e, s[0, 1])


def test_convergence_pulls_nodes_toward_center() -> None:
    f = _field()
    t = 30.0
    x = np.array([-100.0, 100.0])
    y = np.array([0.0, 0.0])
    x1, _ = displaced_positions(f, x, y, t)
    assert abs(x1[0]) < 100.0 and abs(x1[1]) < 100.0, "nodes must move toward the panel centre"


def test_diagonal_is_nan() -> None:
    f = _field()
    x, y = _grid()
    s = strain_matrix(f, x, y, t=10.0)
    assert np.all(np.isnan(np.diag(s))), "self-pairs are not inter-node edges"


def test_factor_is_config_driven() -> None:
    """Default factor must come from config: passing the configured value explicitly
    reproduces the default behaviour exactly (strain is nonlinear in amplitude,
    so this is an equivalence check, not a linearity check)."""
    from src.config import physics_config

    f = _field()
    t = 25.0
    x = np.array([-40.0, 60.0])
    y = np.array([10.0, -5.0])
    s_default = strain_matrix(f, x, y, t)
    expected_factor = float(physics_config()["strain"]["horizontal_displacement_factor"])
    s_explicit = strain_matrix(f, x, y, t, factor=expected_factor)
    assert np.allclose(s_default, s_explicit, atol=1e-15, equal_nan=True), (
        "default strain must equal the strain computed with the config factor"
    )
    # sanity: a different factor must give different strain
    s_other = strain_matrix(f, x, y, t, factor=expected_factor * 2.0)
    assert not np.allclose(s_default, s_other, atol=1e-9)
