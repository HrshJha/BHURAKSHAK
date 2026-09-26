"""T-012 acceptance tests — coupled field identity and config-driven parameters."""

from __future__ import annotations

import numpy as np

from src.config import physics_config
from src.simulator.deformation_field import DeformationField, FieldParams
from src.simulator.spatial_model import spatial_kernel


def _field() -> DeformationField:
    return DeformationField(FieldParams.from_config())


def test_field_equals_product_of_temporal_and_spatial() -> None:
    f = _field()
    rng = np.random.default_rng(7)
    xs = rng.uniform(-200, 200, 12)
    ys = rng.uniform(-200, 200, 12)
    ts = rng.uniform(0, 40, 12)

    got = f(xs, ys, ts)
    expected = np.asarray(f.w_of_t(ts)) * np.asarray(
        spatial_kernel(xs, ys, f.params.panel_center_x, f.params.panel_center_y, f.params.sigma)
    )
    assert np.allclose(got, expected, rtol=0, atol=1e-12), "§10: W(x,y,t) == W(t)·kernel"


def test_field_zero_at_t_zero_everywhere() -> None:
    f = _field()
    xs = np.linspace(-250, 250, 11)
    ys = np.linspace(-250, 250, 11)
    assert np.allclose(f(xs, ys, 0.0), 0.0)


def test_field_at_center_is_w_of_t() -> None:
    f = _field()
    p = f.params
    t = 12.0
    assert np.allclose(f(p.panel_center_x, p.panel_center_y, t), f.w_of_t(t))


def test_params_come_entirely_from_config() -> None:
    """No literal parameters in code: FieldParams must equal configs/physics.yaml."""
    cfg = physics_config()["physics"]
    p = FieldParams.from_config()
    for key, value in cfg.items():
        assert getattr(p, key) == value, f"parameter {key} must be read from config"
    assert p.sigma == cfg["influence_radius"] / physics_config()["kernel"]["sigma_divisor"]


def test_config_change_changes_field_output() -> None:
    """The field is parametrised: different config → different physics."""
    import copy

    cfg = physics_config()
    cfg2 = copy.deepcopy(cfg)
    cfg2["physics"]["maximum_subsidence"] = cfg2["physics"]["maximum_subsidence"]
    cfg2["physics"]["time_coefficient"] = float(cfg["physics"]["time_coefficient"]) * 2.0

    f1 = DeformationField(FieldParams.from_config(cfg))
    f2 = DeformationField(FieldParams.from_config(cfg2))
    t = 10.0
    assert float(f1.w_of_t(t)) != float(f2.w_of_t(t)), "different c must give different W(t)"


def test_derived_w_max_matches_configured_maximum_subsidence() -> None:
    """Geometry-derived amplitude must reconcile with the §10 bookkeeping parameter."""
    f = _field()
    p = f.params
    derived = p.extraction_height * p.subsidence_factor * 1000.0
    assert abs(derived - p.maximum_subsidence) < 1e-6, (
        "extraction_height × subsidence_factor must equal maximum_subsidence (mm) in config"
    )
    assert f.w_max == derived


def test_invalid_config_rejected() -> None:
    import copy

    cfg = physics_config()
    bad = copy.deepcopy(cfg)
    bad["physics"]["time_coefficient"] = -1.0
    try:
        FieldParams.from_config(bad)
    except ValueError:
        return
    raise AssertionError("negative time_coefficient must be rejected")


def test_gradient_matches_finite_difference() -> None:
    """∂W/∂x analytic gradient agrees with central differences of the field itself."""
    f = _field()
    t = 15.0
    h = 1e-3
    xs = np.array([-90.0, -30.0, 0.0, 40.0, 95.0])
    ys = np.array([10.0, -50.0, 70.0, 0.0, 25.0])
    num = (f(xs + h, ys, t) - f(xs - h, ys, t)) / (2 * h)
    ana = f.dW_dx(xs, ys, t)
    assert np.allclose(num, ana, rtol=1e-6, atol=1e-9)
