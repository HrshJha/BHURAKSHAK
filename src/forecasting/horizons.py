"""§16 forecast-horizon resolution (T-078) — NFR-6 discipline.

The five §16 horizons (next-window, 30 min, 1 h, 6 h, 24 h) live in
configs/forecasting.yaml expressed in MINUTES and are resolved here to §10
window-steps using the §9.1 window grid interval
(configs/preprocessing.yaml → resampling.grid_interval_minutes). No horizon
value is hard-coded anywhere under ``src/`` — a different grid or a changed
PRD horizon only requires editing YAML (asserted by
tests/test_forecast_horizons.py).
"""

from __future__ import annotations

from src.config import forecasting_config, load_config

__all__ = [
    "HorizonError",
    "HORIZON_ORDER",
    "configured_horizon_minutes",
    "grid_interval_minutes",
    "horizons_in_steps",
]

#: The five §16 horizons in config-key order — the order forecasts and
#: ``forecast_{channel}_h{steps}`` feature columns are emitted in.
HORIZON_ORDER = ("next_window", "minutes_30", "hours_1", "hours_6", "hours_24")


class HorizonError(ValueError):
    """Raised on an invalid or incomplete §16 horizon configuration."""


def configured_horizon_minutes() -> dict[str, int]:
    """The §16 horizon minutes exactly as configured (all five keys required)."""
    cfg = forecasting_config()["horizons"]
    missing = [k for k in HORIZON_ORDER if k not in cfg]
    if missing:
        raise HorizonError(
            f"configs/forecasting.yaml horizons missing §16 keys: {missing}"
        )
    out: dict[str, int] = {}
    for key in HORIZON_ORDER:
        value = int(cfg[key])
        if value < 0:
            raise HorizonError(f"horizon {key!r} must be >= 0 minutes, got {value}")
        out[key] = value
    return out


def grid_interval_minutes() -> float:
    """The §9.1 window-grid interval the horizons are resolved against."""
    value = float(load_config("preprocessing")["resampling"]["grid_interval_minutes"])
    if value <= 0:
        raise HorizonError("grid_interval_minutes must be > 0")
    return value


def horizons_in_steps(grid_minutes: float | None = None) -> tuple[int, ...]:
    """Resolve the §16 horizons to §10 window-steps (ascending, de-duplicated).

    ``next_window`` (0 minutes in config) is by definition exactly 1 step.
    Every longer horizon must land on the window grid — a multiple of the
    §9.1 grid interval — otherwise the resolution raises rather than silently
    forecasting to a timestamp no window exists for. With the §8.3 default
    10-minute grid the five §16 horizons resolve to (1, 3, 6, 36, 144).
    """
    if grid_minutes is None:
        grid_minutes = grid_interval_minutes()
    minutes = configured_horizon_minutes()
    resolved: list[int] = []
    for key in HORIZON_ORDER:
        m = minutes[key]
        if key == "next_window":
            resolved.append(1)
            continue
        steps = int(round(m / grid_minutes))
        if steps < 1:
            raise HorizonError(
                f"horizon {key!r} ({m} min) is shorter than one "
                f"{grid_minutes:g}-minute window-grid step"
            )
        if abs(steps * grid_minutes - m) > 1e-9:
            raise HorizonError(
                f"horizon {key!r} ({m} min) is not a multiple of the "
                f"{grid_minutes:g}-minute window grid"
            )
        resolved.append(steps)
    return tuple(sorted(set(resolved)))
