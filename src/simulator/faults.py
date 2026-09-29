"""Sensor-fault injection —, (``fault_label``).

Five fault modes, each injectable and each tagged ``SENSOR_FAULT``:

 BIAS — constant offset from onset
 STUCK — readings flatline at the pre-fault level
 DROPOUT — NaN runs (missing data)
 SPIKE — isolated extreme outliers
 DRIFT — slowly growing offset (sensor slow drift)

Without these, Isolation Forest would treat any sensor malfunction as ground
failure. Parameters come from configs/physics.yaml (faults block).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.config import physics_config

FAULT_LABEL = "SENSOR_FAULT"


class FaultType(str, Enum):
    BIAS = "BIAS"
    STUCK = "STUCK"
    DROPOUT = "DROPOUT"
    SPIKE = "SPIKE"
    DRIFT = "DRIFT"


@dataclass(frozen=True)
class FaultParams:
    bias_mm: float
    drift_mm_per_day: float
    dropout_probability: float
    dropout_run_length: int
    spike_probability: float
    spike_magnitude_mm: float
    stuck_duration_steps: int

    @classmethod
    def from_config(cls) -> "FaultParams":
        f = physics_config()["faults"]
        return cls(
            bias_mm=float(f["bias_mm"]),
            drift_mm_per_day=float(f["drift_mm_per_day"]),
            dropout_probability=float(f["dropout_probability"]),
            dropout_run_length=int(f["dropout_run_length"]),
            spike_probability=float(f["spike_probability"]),
            spike_magnitude_mm=float(f["spike_magnitude_mm"]),
            stuck_duration_steps=int(f["stuck_duration_steps"]),
        )


@dataclass
class FaultResult:
    values: np.ndarray        # corrupted series (NaN where DROPOUT)
    fault_mask: np.ndarray    # True where the sample is fault-affected
    fault_type: FaultType


def inject_fault(
    values: np.ndarray,
    fault_type: FaultType,
    rng: np.random.Generator,
    onset_step: int,
    days_per_step: float = 1.0,
    params: FaultParams | None = None,
) -> FaultResult:
    """Corrupt ``values`` from ``onset_step`` onward with the given fault mode.

 Returns the corrupted series plus a boolean mask marking fault-affected
 samples — the mask drives the ``SENSOR_FAULT`` label so faulted
 rows are separable from genuine ground movement.
 """
    p = params or FaultParams.from_config()
    v = np.array(values, dtype=float)
    n = v.size
    if not 0 <= onset_step < n:
        raise ValueError("onset_step out of range")
    mask = np.zeros(n, dtype=bool)
    mask[onset_step:] = True

    if fault_type is FaultType.BIAS:
        v[onset_step:] += p.bias_mm
    elif fault_type is FaultType.STUCK:
        stuck_value = v[onset_step]
        end = min(onset_step + p.stuck_duration_steps, n)
        v[onset_step:end] = stuck_value
        mask[onset_step:end] = True
        mask[end:] = False  # sensor recovers; only the stuck window is fault-affected
    elif fault_type is FaultType.DROPOUT:
        i = onset_step
        while i < n:
            if rng.random() < p.dropout_probability:
                run_len = min(p.dropout_run_length, n - i)
                v[i : i + run_len] = np.nan
                i += run_len
            else:
                i += 1
    elif fault_type is FaultType.SPIKE:
        spike_idx = np.arange(onset_step, n)[rng.random(n - onset_step) < p.spike_probability]
        direction = rng.choice([-1.0, 1.0], size=spike_idx.size)
        v[spike_idx] += direction * p.spike_magnitude_mm
        mask[:] = False
        mask[spike_idx] = True
    elif fault_type is FaultType.DRIFT:
        steps = np.arange(n - onset_step, dtype=float)
        v[onset_step:] += p.drift_mm_per_day * days_per_step * steps
    else:  # pragma: no cover — enum exhaustiveness
        raise ValueError(f"unknown fault type {fault_type}")

    return FaultResult(values=v, fault_mask=mask, fault_type=fault_type)


def all_fault_types() -> list[FaultType]:
    """The five fault modes — satisfies the requirement of ≥3 fault types."""
    return list(FaultType)
