"""T-019 acceptance tests — exactly 400 config-driven virtual nodes."""

from __future__ import annotations

import numpy as np

from src.config import physics_config
from src.simulator.grid import build_grid


def test_exactly_400_nodes() -> None:
    grid = build_grid()
    assert grid.n_nodes == 400, "§10: 20×20 = 400 virtual nodes"
    assert len(grid.node_ids) == 400
    assert len(set(grid.node_ids)) == 400, "node_ids must be unique"


def test_grid_dimensions_config_driven() -> None:
    cfg = physics_config()
    n_side = int(cfg["grid"]["nodes_per_side"])
    grid = build_grid()
    xs = np.unique(grid.x)
    ys = np.unique(grid.y)
    assert xs.size == n_side and ys.size == n_side


def test_spacing_matches_config() -> None:
    cfg = physics_config()
    spacing = float(cfg["grid"]["spacing_m"])
    grid = build_grid()
    xs = np.unique(grid.x)
    assert np.allclose(np.diff(xs), spacing)


def test_grid_centred_on_origin() -> None:
    grid = build_grid()
    assert np.isclose(grid.x.min(), -grid.x.max())
    assert np.isclose(grid.y.mean(), 0.0) and np.isclose(grid.x.mean(), 0.0)


def test_custom_config_changes_grid() -> None:
    import copy

    from src.config import physics_config

    cfg = copy.deepcopy(physics_config())
    cfg["grid"]["nodes_per_side"] = 5
    cfg["grid"]["spacing_m"] = 10.0
    grid = build_grid(cfg)
    assert grid.n_nodes == 25, "grid size must follow config, not hardcode"
    assert np.isclose(np.unique(grid.x)[1] - np.unique(grid.x)[0], 10.0)


def test_invalid_config_raises() -> None:
    import copy

    from src.config import physics_config

    bad = copy.deepcopy(physics_config())
    bad["grid"]["nodes_per_side"] = 0
    try:
        build_grid(bad)
    except ValueError:
        return
    raise AssertionError("nodes_per_side=0 must raise")
