"""Model registry — PRD §30, FR-14, §10.1 (T-055).

FR-14: the system shall log every prediction with **model name, version,
feature-schema version and training-dataset version** for traceability, plus
a timestamp (§30). This module makes that requirement structural:

- :meth:`ModelRegistry.register_model` stores a model entry under
  ``models/registry.json`` and REFUSES versions of the training dataset that
  do not resolve to an existing §10.1 manifest (the hard requirement of
  §10.1: "without it, … model-registry traceability cannot actually be
  verified after the fact"). A feature-version mismatch with the manifest's
  ``feature_schema_version`` is likewise rejected;
- :meth:`ModelRegistry.log_prediction` emits prediction records that carry
  the five FR-14 fields BY CONSTRUCTION — they are copied from the registered
  entry, so a logged prediction cannot omit them;
- prediction records are append-only JSON Lines; the registry itself is JSON
  and reloaded on construction, so entries survive the session.

Default locations follow §30's artifact layout: the registry lives at
``models/registry.json`` and dataset manifests are resolved from
``data/synthetic/dataset_manifest.json`` (§10.1).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["ModelRegistryError", "ModelEntry", "ModelRegistry", "REPO_ROOT", "DEFAULT_REGISTRY_PATH", "DEFAULT_MANIFEST_PATHS"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"
DEFAULT_MANIFEST_PATHS: tuple[Path, ...] = (
    REPO_ROOT / "data" / "synthetic" / "dataset_manifest.json",
    REPO_ROOT / "data" / "recorded" / "tabletop" / "dataset_manifest.json",
)

#: the FR-14 traceability fields every logged prediction must carry
FR14_FIELDS = ("model_name", "model_version", "feature_version", "training_dataset_version", "timestamp")


class ModelRegistryError(ValueError):
    """Raised on invalid registry entries, manifests or prediction logs."""


@dataclass
class ModelEntry:
    """One registered model: identity + the §10.1 dataset it was trained on."""

    model_name: str
    model_version: str
    feature_version: str
    training_dataset_version: str
    registered_at: str
    artifact_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "training_dataset_version": self.training_dataset_version,
            "registered_at": self.registered_at,
        }
        if self.artifact_path is not None:
            out["artifact_path"] = self.artifact_path
        return out


class ModelRegistry:
    """§30 registry: model entries in JSON, prediction log as append-only JSONL."""

    def __init__(
        self,
        path: str | Path | None = None,
        manifest_paths: list[str | Path] | None = None,
        predictions_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
        self.manifest_paths = [Path(p) for p in (manifest_paths if manifest_paths is not None else DEFAULT_MANIFEST_PATHS)]
        self.predictions_path = Path(predictions_path) if predictions_path is not None else self.path.parent / "predictions.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: dict[str, ModelEntry] = {}
        self._load()

    # --- registry entries ---------------------------------------------------------

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for raw in data.get("models", []):
                entry = ModelEntry(
                    model_name=raw["model_name"],
                    model_version=raw["model_version"],
                    feature_version=raw["feature_version"],
                    training_dataset_version=raw["training_dataset_version"],
                    registered_at=raw["registered_at"],
                    artifact_path=raw.get("artifact_path"),
                )
                self._entries[self._key(entry.model_name, entry.model_version)] = entry

    @staticmethod
    def _key(model_name: str, model_version: str) -> str:
        return f"{model_name}@{model_version}"

    def _flush(self) -> None:
        payload = {"models": [e.to_dict() for e in self._entries.values()]}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # --- §10.1 manifest resolution --------------------------------------------------

    def _manifests(self) -> dict[str, tuple[dict[str, Any], Path]]:
        found: dict[str, tuple[dict[str, Any], Path]] = {}
        for p in self.manifest_paths:
            if p.exists():
                manifest = json.loads(p.read_text(encoding="utf-8"))
                version = manifest.get("dataset_version")
                if not version:
                    raise ModelRegistryError(f"manifest {p} lacks a dataset_version field (§10.1)")
                found[str(version)] = (manifest, p)
        return found

    def resolve_training_dataset(self, version: str) -> tuple[dict[str, Any], Path]:
        """§10.1: the dataset version must resolve to an EXISTING manifest."""
        found = self._manifests()
        if str(version) not in found:
            available = sorted(found) or "none (no manifest files exist)"
            raise ModelRegistryError(
                f"training_dataset_version {version!r} does not resolve to an existing §10.1 manifest "
                f"(available: {available})"
            )
        return found[str(version)]

    # --- registration ----------------------------------------------------------------

    def register_model(
        self,
        *,
        model_name: str,
        model_version: str,
        feature_version: str,
        training_dataset_version: str,
        artifact_path: str | None = None,
        now: datetime | None = None,
    ) -> ModelEntry:
        """Register a model; the dataset version must resolve (§10.1/§30)."""
        if not str(model_name).strip() or not str(model_version).strip():
            raise ModelRegistryError("model_name and model_version are required")
        manifest, _ = self.resolve_training_dataset(training_dataset_version)
        schema_version = manifest.get("feature_schema_version")
        if schema_version is not None and str(schema_version) != str(feature_version):
            raise ModelRegistryError(
                f"feature_version {feature_version!r} does not match the dataset manifest's "
                f"feature_schema_version {schema_version!r}"
            )
        ts = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        entry = ModelEntry(
            model_name=str(model_name),
            model_version=str(model_version),
            feature_version=str(feature_version),
            training_dataset_version=str(training_dataset_version),
            registered_at=ts,
            artifact_path=artifact_path,
        )
        with self._lock:
            self._entries[self._key(entry.model_name, entry.model_version)] = entry
            self._flush()
        return entry

    def get_entry(self, model_name: str, model_version: str) -> ModelEntry:
        key = self._key(model_name, model_version)
        if key not in self._entries:
            raise ModelRegistryError(f"model {key!r} is not registered — register it before logging predictions")
        return self._entries[key]

    def list_models(self) -> list[ModelEntry]:
        return list(self._entries.values())

    # --- FR-14 prediction log ----------------------------------------------------------

    def log_prediction(
        self,
        model_name: str,
        model_version: str,
        payload: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Append one prediction record carrying the five FR-14 fields by construction.

        ``payload`` holds the prediction-specific content (node_id,
        probabilities, alert level, …); the traceability fields are copied
        from the registered entry so they can never be missing or diverge.
        """
        entry = self.get_entry(model_name, model_version)
        ts = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        record: dict[str, Any] = {
            "model_name": entry.model_name,
            "model_version": entry.model_version,
            "feature_version": entry.feature_version,
            "training_dataset_version": entry.training_dataset_version,
            "timestamp": ts,
        }
        record.update(payload)
        with self._lock:
            self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
            with self.predictions_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def read_predictions(self) -> list[dict[str, Any]]:
        if not self.predictions_path.exists():
            return []
        with self.predictions_path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
