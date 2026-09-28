"""T-007 — config-loader tests including the NFR-6 no-hardcoded-threshold scan.

The scan asserts the §21.1 risk-threshold literals (0.5, 0.6, 0.7, 1.5) appear
nowhere under ``src/`` outside the loader itself, so risk behaviour can only
come from ``configs/alerts.yaml``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config import (
    CONFIG_DIR,
    REPO_ROOT,
    alerts_config,
    deescalation_multiplier,
    escalation_thresholds,
    load_config,
    physics_parameters,
    sampling_config,
)

SRC_DIR = REPO_ROOT / "src"
LOADER_PATH = SRC_DIR / "config.py"

# §21.1 threshold literals, matched as standalone numeric literals only.
THRESHOLD_LITERALS = ("0.5", "0.6", "0.7", "1.5")


def _literal_pattern(literal: str) -> re.Pattern[str]:
    """Match the literal as a bare float, not inside 10.5, 0.55, or 'v0.5'."""
    return re.compile(rf"(?<![\w.]){re.escape(literal)}(?![\w.])")


def test_all_expected_config_files_exist() -> None:
    for expected in (
        "sampling.yaml",
        "alerts.yaml",
        "physics.yaml",
        "feature_schema_v1.yaml",
        "preprocessing.yaml",
    ):
        assert (CONFIG_DIR / expected).is_file(), f"missing config file: {expected}"


def test_load_config_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        load_config("does_not_exist")


def test_loader_returns_alerts_thresholds() -> None:
    esc = escalation_thresholds()
    assert set(esc.keys()) == {"GREEN_to_WATCH", "WATCH_to_WARNING", "WARNING_to_CRITICAL"}
    assert alerts_config()["risk_levels"] == ["GREEN", "WATCH", "WARNING", "CRITICAL"]


def test_loader_cached() -> None:
    assert load_config("physics") is load_config("physics"), "load_config must be cached"


def test_physics_parameters_are_floats() -> None:
    params = physics_parameters()
    assert len(params) == 10, "§10 defines exactly ten physics parameters"
    assert all(isinstance(v, float) for v in params.values())


def test_sampling_config_reachable_through_loader() -> None:
    assert "channels" in sampling_config()


def test_velocity_and_acceleration_units_are_explicit() -> None:
    units = load_config("feature_schema")["units"]
    assert units["B_temporal"]["velocity"] == "mm/hour"
    assert units["B_temporal"]["acceleration"] == "mm/hour^2"
    assert units["G_dgps"]["velocity"] == "mm/year"
    assert units["G_dgps"]["acceleration"] == "mm/year^2"


def test_preprocessing_config_reachable_through_loader() -> None:
    prep = load_config("preprocessing")
    assert prep["windowing"]["window_steps"] == 60  # §10 windowing
    assert prep["windowing"]["stride"] == 10
    assert prep["align_modalities"]["insar_tolerance_hours"] == 12.0  # §9.1
    assert prep["align_modalities"]["dgps_tolerance_hours"] == 1.0


def test_no_hardcoded_risk_thresholds_outside_loader() -> None:
    """NFR-6 scan: §21.1 literals must not appear as constants in src/ beyond the loader."""
    offenders: list[str] = []
    py_files = sorted(SRC_DIR.rglob("*.py"))
    assert py_files, "src/ must contain python sources for the scan to be meaningful"

    for path in py_files:
        if path.resolve() == LOADER_PATH.resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for literal in THRESHOLD_LITERALS:
            if _literal_pattern(literal).search(text):
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains literal {literal}")

    assert not offenders, (
        "NFR-6 violation — risk-threshold literals must live in configs/, not src/:\n  "
        + "\n  ".join(offenders)
    )
