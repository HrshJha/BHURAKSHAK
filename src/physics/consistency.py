"""Compare observed displacement and tilt with the configured deformation model."""

from __future__ import annotations

import numpy as np

from src.simulator.deformation_field import DeformationField

__all__ = ["PhysicsConsistency", "physics_engine"]


class PhysicsConsistency:
    """Wrapper that reuses the field for expected deformation."""

    def __init__(self, field: DeformationField | None = None) -> None:
        self.field = field or DeformationField()

    def expected_displacement_mm(
        self, x: float | np.ndarray, y: float | np.ndarray, t_hours: float | np.ndarray
    ) -> np.ndarray:
        """Expected subsidence (mm) at mesh-local (x, y) at time t (hours)."""
        return np.asarray(self.field.expected_displacement(x, y, np.asarray(t_hours) / 24.0))

    def expected_tilt(
        self, x: float | np.ndarray, y: float | np.ndarray, t_hours: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expected (tilt_x, tilt_y) from the analytic gradient of the model."""
        tx, ty = self.field.expected_tilt(x, y, np.asarray(t_hours) / 24.0)
        return np.asarray(tx), np.asarray(ty)

    def displacement_residual(
        self,
        observed_mm: float | np.ndarray,
        x: float | np.ndarray,
        y: float | np.ndarray,
        t_hours: float | np.ndarray,
    ) -> np.ndarray:
        """: physics_residual = observed_deformation − expected_deformation."""
        return np.asarray(observed_mm, dtype=float) - self.expected_displacement_mm(x, y, t_hours)

    def tilt_residual(
        self,
        observed_tilt_x: float | np.ndarray,
        observed_tilt_y: float | np.ndarray,
        x: float | np.ndarray,
        y: float | np.ndarray,
        t_hours: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-axis residual between observed and expected tilt."""
        etx, ety = self.expected_tilt(x, y, t_hours)
        return np.asarray(observed_tilt_x, dtype=float) - etx, np.asarray(observed_tilt_y, dtype=float) - ety


def physics_engine() -> PhysicsConsistency:
    """Config-built engine ( parameters from configs/physics.yaml)."""
    return PhysicsConsistency()
