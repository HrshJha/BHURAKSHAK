"""20×20 virtual-grid mesh generator — PRD §10 (T-019).

Produces the virtual node layout for the synthetic dataset: exactly
``nodes_per_side²`` nodes (§10 scale: 20×20 = 400) with mesh-local (x, y)
coordinates in metres, config-driven from configs/physics.yaml (grid block;
§9.2 mesh-local frame). The generator is designed so scale can grow later
(§10) — only the config changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import physics_config


@dataclass(frozen=True)
class MeshGrid:
    node_ids: list[str]
    x: np.ndarray
    y: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.x.size)


def build_grid(cfg: dict | None = None) -> MeshGrid:
    """Build the virtual node mesh: centred square grid, uniform spacing."""
    cfg = cfg or physics_config()
    grid = cfg["grid"]
    n_side = int(grid["nodes_per_side"])
    spacing = float(grid["spacing_m"])
    if n_side <= 0 or spacing <= 0:
        raise ValueError("grid.nodes_per_side and grid.spacing_m must be > 0")

    coords = (np.arange(n_side) - (n_side - 1) / 2.0) * spacing
    xx, yy = np.meshgrid(coords, coords)
    x = xx.ravel()
    y = yy.ravel()
    node_ids = [f"V{i:04d}" for i in range(x.size)]
    return MeshGrid(node_ids=node_ids, x=x, y=y)
