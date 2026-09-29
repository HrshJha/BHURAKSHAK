"""Configuration loader for the BhuRakshak Data+ML workstream.

 refs: (all risk thresholds, sampling rates and alert escalation
rules must be configuration-driven, not hard-coded), (reproducible,
versioned parameters).

Every consumer in ``src/`` must obtain operational thresholds, sampling
rates and physics parameters through this module. The repo-wide scan in
``tests/test_config_loader.py`` asserts that the risk-threshold
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
    "feature_schema": "feature_schema_v2.yaml",
    "features": "features.yaml",
    "feature_provenance": "feature_provenance.yaml",
    "preprocessing": "preprocessing.yaml",
    "anomaly": "anomaly.yaml",
    "validation": "validation.yaml",
    "risk_model": "risk_model.yaml",
    "environmental": "environmental.yaml",
    "forecasting": "forecasting.yaml",
    "model_params": "model_params.yaml",
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
    """ sampling-rate defaults."""
    return load_config("sampling")


def alerts_config() -> dict[str, Any]:
    """ alert-engine escalation/de-escalation config."""
    return load_config("alerts")


def physics_config() -> dict[str, Any]:
    """ physics/simulator parameters."""
    return load_config("physics")


def feature_schema_config() -> dict[str, Any]:
    """ / current feature schema."""
    return load_config("feature_schema")


def anomaly_config() -> dict[str, Any]:
    """ Isolation Forest parameters + ablation sets."""
    return load_config("anomaly")


def validation_config() -> dict[str, Any]:
    """ split fractions."""
    return load_config("validation")


def risk_model_config() -> dict[str, Any]:
    """ / risk-model parameters (…)."""
    return load_config("risk_model")


def environmental_config() -> dict[str, Any]:
    """Feature Group J gate + synthetic source parameters."""
    return load_config("environmental")


def forecasting_config() -> dict[str, Any]:
    """ temporal-forecasting parameters."""
    return load_config("forecasting")


def model_params_config() -> dict[str, Any]:
    """Tuned model parameters and provenance."""
    return load_config("model_params")


def escalation_thresholds() -> dict[str, dict[str, Any]]:
    """The three escalation transitions exactly as configured."""
    esc = alerts_config()["escalation"]
    return {
        "GREEN_to_WATCH": esc["GREEN_to_WATCH"],
        "WATCH_to_WARNING": esc["WATCH_to_WARNING"],
        "WARNING_to_CRITICAL": esc["WARNING_to_CRITICAL"],
    }


def deescalation_multiplier() -> float:
    """ de-escalation hysteresis multiplier (default 1.5)."""
    return float(alerts_config()["de_escalation"]["persistence_multiplier"])


def physics_parameters() -> dict[str, float]:
    """The ten physics parameters (validated present by test_configs)."""
    return {k: float(v) for k, v in physics_config()["physics"].items()}
