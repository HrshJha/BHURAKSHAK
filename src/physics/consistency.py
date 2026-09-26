"""Physics engine — physics_residual — PRD §21, FR-15 (T-040).

§21 (via FR-15): the pipeline runs a physics-consistency check,
``physics_residual = observed_deformation − expected_deformation``, where the
expected deformation comes from the **same §10 influence-function/Knothe
model that generated the synthetic data** — reused, never re-implemented
(one source of physical truth, per §10's "physical coupling" principle).

The residual is the "is the ground behaving like the model says?" signal:
≈ 0 when observation equals the model; structurally large when the observed
movement cannot be explained by the configured mining geometry. It feeds
XGBoost as a feature (Group F, T-041) and the explainability breakdown (§22).

Units/timescales: the raw node table stamps rows in decimal **hours**; the
Knothe time base is **days** (configs/physics.yaml ``time_coefficient`` per
day). This module owns that conversion so callers never guess.
"""

from __future__ import annotations

import numpy as np

from src.simulator.deformation_field import DeformationField

__all__ = ["PhysicsConsistency", "physics_engine"]


class PhysicsConsistency:
    """Wrapper that reuses the §10 field for expected deformation (§21)."""

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
        """Expected (tilt_x, tilt_y) from the analytic gradient of the §10 model."""
        tx, ty = self.field.expected_tilt(x, y, np.asarray(t_hours) / 24.0)
        return np.asarray(tx), np.asarray(ty)

    def displacement_residual(
        self,
        observed_mm: float | np.ndarray,
        x: float | np.ndarray,
        y: float | np.ndarray,
        t_hours: float | np.ndarray,
    ) -> np.ndarray:
        """§21: physics_residual = observed_deformation − expected_deformation."""
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
    """Config-built engine (§10 parameters from configs/physics.yaml)."""
    return PhysicsConsistency()
