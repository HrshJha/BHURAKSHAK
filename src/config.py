"""Configuration loader for the SubSense Data+ML workstream.

PRD refs: NFR-6 (all risk thresholds, sampling rates and alert escalation
rules must be configuration-driven, not hard-coded), NFR-7 (reproducible,
versioned parameters).

Every consumer in ``src/`` must obtain operational thresholds, sampling
rates and physics parameters through this module. The repo-wide scan in
``tests/test_config_loader.py`` asserts that the §21.1 risk-threshold
literals appear nowhere under ``src/`` outside this loader.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

_CONFIG_FILES = {
    "sampling": "sampling.yaml",
    "alerts": "alerts.yaml",
    "physics": "physics.yaml",
    "feature_schema": "feature_schema_v1.yaml",
    "preprocessing": "preprocessing.yaml",
    "anomaly": "anomaly.yaml",
}


@lru_cache(maxsize=None)
def load_config(name: str) -> dict[str, Any]:
    """Load one config file by logical name, cached.

    Raises ``KeyError`` for an unknown name and ``FileNotFoundError`` if the
    underlying YAML file is missing — callers fail loudly, never silently on
    defaults.
    """
    if name not in _CONFIG_FILES:
        raise KeyError(f"unknown config name {name!r}; expected one of {sorted(_CONFIG_FILES)}")
    path = CONFIG_DIR / _CONFIG_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(f"required config file missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must contain a YAML mapping at top level")
    return data


def sampling_config() -> dict[str, Any]:
    """PRD §8.3 sampling-rate defaults (T-003)."""
    return load_config("sampling")


def alerts_config() -> dict[str, Any]:
    """PRD §21.1 alert-engine escalation/de-escalation config (T-004)."""
    return load_config("alerts")


def physics_config() -> dict[str, Any]:
    """PRD §10 physics/simulator parameters (T-005)."""
    return load_config("physics")


def feature_schema_config() -> dict[str, Any]:
    """PRD §11/§13 feature schema v1 (T-006)."""
    return load_config("feature_schema")


def anomaly_config() -> dict[str, Any]:
    """PRD §14 Isolation Forest parameters + ablation sets (T-043/T-045)."""
    return load_config("anomaly")


def escalation_thresholds() -> dict[str, dict[str, Any]]:
    """The three §21.1 escalation transitions exactly as configured."""
    esc = alerts_config()["escalation"]
    return {
        "GREEN_to_WATCH": esc["GREEN_to_WATCH"],
        "WATCH_to_WARNING": esc["WATCH_to_WARNING"],
        "WARNING_to_CRITICAL": esc["WARNING_to_CRITICAL"],
    }


def deescalation_multiplier() -> float:
    """§21.1 de-escalation hysteresis multiplier (default 1.5)."""
    return float(alerts_config()["de_escalation"]["persistence_multiplier"])


def physics_parameters() -> dict[str, float]:
    """The ten §10 physics parameters (validated present by test_configs)."""
    return {k: float(v) for k, v in physics_config()["physics"].items()}
