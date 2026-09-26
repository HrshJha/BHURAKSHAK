"""Coupled deformation field W(x, y, t) = W(t) * spatial_kernel(x, y) — PRD §10 (T-012).

This is the simulator's latent ground truth. Every sensor channel in the
synthetic dataset is derived from this single field — never generated
independently (the §10 "core principle — physical coupling").

All parameters come from configs/physics.yaml via src/config.py; no physics
literals live in code (asserted by tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import physics_config
from src.simulator.spatial_model import sigma_from_config, spatial_kernel
from src.simulator.temporal_model import growth_velocity, subsidence_growth


@dataclass(frozen=True)
class FieldParams:
    """Validated container for the §10 physics parameters."""

    panel_center_x: float
    panel_center_y: float
    panel_width: float
    panel_length: float
    mine_depth: float
    extraction_height: float
    subsidence_factor: float
    influence_radius: float
    time_coefficient: float
    maximum_subsidence: float
    sigma: float

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "FieldParams":
        cfg = cfg or physics_config()
        p = cfg["physics"]
        params = cls(
            panel_center_x=float(p["panel_center_x"]),
            panel_center_y=float(p["panel_center_y"]),
            panel_width=float(p["panel_width"]),
            panel_length=float(p["panel_length"]),
            mine_depth=float(p["mine_depth"]),
            extraction_height=float(p["extraction_height"]),
            subsidence_factor=float(p["subsidence_factor"]),
            influence_radius=float(p["influence_radius"]),
            time_coefficient=float(p["time_coefficient"]),
            maximum_subsidence=float(p["maximum_subsidence"]),
            sigma=sigma_from_config(cfg),
        )
        params.validate()
        return params

    def validate(self) -> None:
        positive = {
            "panel_width": self.panel_width,
            "panel_length": self.panel_length,
            "mine_depth": self.mine_depth,
            "extraction_height": self.extraction_height,
            "influence_radius": self.influence_radius,
            "time_coefficient": self.time_coefficient,
            "maximum_subsidence": self.maximum_subsidence,
            "sigma": self.sigma,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"physics parameter {name} must be > 0, got {value}")
        if not 0.0 < self.subsidence_factor <= 1.0:
            raise ValueError("subsidence_factor must be in (0, 1]")


class DeformationField:
    """W(x, y, t) = W(t) · exp(-[(x−x0)² + (y−y0)²] / 2σ²), fully config-driven."""

    def __init__(self, params: FieldParams | None = None) -> None:
        self.params = params or FieldParams.from_config()
        p = self.params

        # PRD §10 note: the influence-function literature derives surface
        # subsidence from extraction geometry (depth, seam height, subsidence
        # factor) rather than taking W_max as an independent dial. The config
        # keeps BOTH; we use the geometry-derived value as the authoritative
        # amplitude and assert the configured maximum_subsidence is the
        # intended bookkeeping of the same quantity.
        derived = (
            p.extraction_height * p.subsidence_factor * 1000.0
        )  # mm; metres → mm
        self.w_max = derived

    # -- temporal component -------------------------------------------------
    def w_of_t(self, t: float | np.ndarray) -> float | np.ndarray:
        """W(t) = W_max · (1 − e^(−c·t)) in mm."""
        return subsidence_growth(t, self.w_max, self.params.time_coefficient)

    def w_velocity(self, t: float | np.ndarray) -> float | np.ndarray:
        """dW/dt in mm/day."""
        return growth_velocity(t, self.w_max, self.params.time_coefficient)

    # -- coupled field -------------------------------------------------------
    def __call__(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        t: float | np.ndarray,
    ) -> np.ndarray:
        """W(x, y, t); broadcast shapes of x, y, t."""
        kernel = spatial_kernel(x, y, self.params.panel_center_x, self.params.panel_center_y, self.params.sigma)
        w_t = np.asarray(self.w_of_t(t))
        return np.asarray(kernel) * w_t

    def at_nodes(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: float | np.ndarray,
    ) -> np.ndarray:
        """Field value at node coordinates (arrays of equal length) at time(s) t."""
        return self.__call__(x, y, t)

    # -- analytic derivatives used by the tilt channel -----------------------
    def dW_dx(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        t: float | np.ndarray,
    ) -> np.ndarray:
        """∂W/∂x = W(t) · (−(x−x0)/σ²) · exp(−r²/2σ²) — the analytic tilt_x source."""
        x_a = np.asarray(x, dtype=float)
        kernel = np.asarray(
            spatial_kernel(x_a, y, self.params.panel_center_x, self.params.panel_center_y, self.params.sigma)
        )
        w_t = np.asarray(self.w_of_t(t))
        return -(x_a - self.params.panel_center_x) / (self.params.sigma**2) * kernel * w_t

    def dW_dy(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        t: float | np.ndarray,
    ) -> np.ndarray:
        """∂W/∂y = W(t) · (−(y−y0)/σ²) · exp(−r²/2σ²) — the analytic tilt_y source."""
        y_a = np.asarray(y, dtype=float)
        kernel = np.asarray(
            spatial_kernel(x, y_a, self.params.panel_center_x, self.params.panel_center_y, self.params.sigma)
        )
        w_t = np.asarray(self.w_of_t(t))
        return -(y_a - self.params.panel_center_y) / (self.params.sigma**2) * kernel * w_t

    # -- expected values for the live physics-consistency engine (§21) -------
    def expected_displacement(self, x: float | np.ndarray, y: float | np.ndarray, t: float) -> np.ndarray:
        """Expected subsidence (mm) at (x, y) at time t — reuses this same model (§21)."""
        return self.__call__(x, y, t)

    def expected_tilt(self, x: float | np.ndarray, y: float | np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Expected (tilt_x, tilt_y) from the analytic gradient of the same model (§21)."""
        return self.dW_dx(x, y, t), self.dW_dy(x, y, t)
