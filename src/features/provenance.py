"""Feature-source registry and deployment allow-list."""

from __future__ import annotations

from typing import Iterable
import re

from src.config import load_config

FORBIDDEN_SOURCE_NAMES = frozenset(
    {"anomaly_label", "fault_label", "risk_label", "data_quality_label", "scenario_family", "scenario_type", "event_id", "sequence_id"}
)


class FeatureProvenanceError(ValueError):
    """Raised when a feature has unknown or forbidden provenance."""


def registry() -> dict[str, dict]:
    return load_config("feature_provenance")["features"]


def _entry(name: str) -> dict | None:
    cfg = load_config("feature_provenance")
    if name in cfg["features"]:
        return cfg["features"][name]
    for pattern in cfg.get("patterns", []):
        if re.fullmatch(pattern["pattern"].removeprefix("^").removesuffix("$"), name):
            return pattern
    return None


def assert_registered(features: Iterable[str]) -> None:
    entries = registry()
    missing = sorted(name for name in set(features) if _entry(name) is None)
    if missing:
        raise FeatureProvenanceError(f"features missing provenance entries: {missing}")
    for name in features:
        entry = _entry(name)
        klass = entry.get("class")
        if klass not in {"observable", "config_derived", "oracle", "label_derived"}:
            raise FeatureProvenanceError(f"{name!r} has invalid provenance class {klass!r}")
        sources = set(entry.get("sources", []))
        forbidden = sorted(
            source
            for source in sources
            if source in FORBIDDEN_SOURCE_NAMES
            or source.startswith(("injected_", "true_"))
            or source.endswith(("_truth", "_label"))
        )
        if forbidden:
            raise FeatureProvenanceError(f"{name!r} uses forbidden source columns {forbidden}")


def assert_group_registered(group: str, features: Iterable[str]) -> None:
    """Validate an emitter's names with group-qualified disambiguation."""
    entries = registry()
    sources = []
    missing = []
    for name in features:
        qualified = f"{group}.{name}"
        if qualified in entries:
            sources.append(qualified)
        elif name in entries:
            sources.append(name)
        else:
            missing.append(qualified)
    if missing:
        raise FeatureProvenanceError(f"emitted features missing group provenance entries: {sorted(missing)}")
    assert_registered(sources)


def model_input_allowlist(candidates: Iterable[str] | None = None) -> set[str]:
    """Return registered observable/config-derived features not gated by schema."""
    if candidates is None:
        candidates = registry()
    assert_registered(candidates)
    return {
        name
        for name in candidates
        if (_entry(name) or {}).get("class") in set(load_config("feature_provenance")["allowed_classes"])
        and not (_entry(name) or {}).get("gated", False)
    }
