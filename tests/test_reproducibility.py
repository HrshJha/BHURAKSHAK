""" acceptance tests — same seed ⇒ byte-identical output; different seed ⇒ not."""

from __future__ import annotations

import hashlib

import numpy as np

from src.simulator.grid import build_grid
from src.simulator.rng import make_rng
from src.simulator.scenarios import Scenario, generate_scenario


def _run_digest(seed: int) -> str:
    """Generate a representative slice of dataset output and hash the bytes."""
    grid = build_grid()
    rng = make_rng(seed)
    parts: list[bytes] = []
    # deterministic node subset: first 8 nodes across a spread of scenarios
    for i in range(8):
        x = float(grid.x[i])
        y = float(grid.y[i])
        scenario = list(Scenario)[i % len(Scenario)]
        res = generate_scenario(scenario, x, y, grid.node_ids[i], rng)
        for arr in (
            res.timestamps_hours, res.subsidence_mm, res.tilt_x_deg, res.tilt_y_deg,
            res.displacement_mm, res.strain, res.vibration_rms, res.battery_v,
        ):
            parts.append(np.nan_to_num(arr, nan=-999.0).tobytes())
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def test_same_seed_byte_identical() -> None:
    assert _run_digest(42) == _run_digest(42), "same seed must reproduce byte-identical output"


def test_different_seed_differs() -> None:
    assert _run_digest(42) != _run_digest(43), "different seeds must produce different output"


def test_rng_factory_deterministic() -> None:
    r1 = make_rng(7)
    r2 = make_rng(7)
    a1 = [r1.normal() for _ in range(100)]
    a2 = [r2.normal() for _ in range(100)]
    assert a1 == a2
