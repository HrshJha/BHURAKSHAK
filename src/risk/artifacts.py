"""Versioned model artifacts with integrity + metadata (fixall Phase 2).

Closes audit MEDIUM (artifacts carry no version/schema) and MEDIUM
(unauthenticated pickle). Every saved bundle embeds an block

 model_name, model_version, feature_version, training_dataset_version,
 timestamp, split_name, seed

plus ``preprocessing`` (what the loader must apply before the model sees
data: ``"standardise"`` for bundles carrying a fitted scaler, ``"none"`` for
scale-invariant models). A sha256 sidecar (``<path>.sha256``) is written at
save time and verified at load time — a tampered or truncated file fails
loudly instead of silently deserialising.

Saves register through:mod:`src.risk.model_registry` when the dataset
version resolves, so ``models/registry.json`` is the single index of what is
shipped and every prediction path can log records from a registered
entry.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import feature_schema_config
from src.risk.model_registry import ModelRegistry, ModelRegistryError

__all__ = ["ArtifactError", "save_model_artifact", "load_model_artifact"]

FR14_PAYLOAD_FIELDS = (
    "model_name",
    "model_version",
    "feature_version",
    "training_dataset_version",
    "timestamp",
    "split_name",
    "seed",
)


class ArtifactError(RuntimeError):
    """Raised on artifact save/load contract violations or integrity failures."""


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _registry() -> ModelRegistry:
    return ModelRegistry()


def save_model_artifact(
    path: str | Path,
    *,
    model: Any,
    features: list[str],
    model_name: str,
    model_version: str,
    training_dataset_version: str,
    split_name: str,
    seed: int,
    scaler: Any | None = None,
    preprocessing: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dump a versioned bundle + sha256 sidecar, and register it.

 ``preprocessing`` defaults to ``"standardise"`` when a ``scaler`` is
 given and ``"none"`` otherwise; passing both a scaler and
 ``preprocessing="none"`` raises (the declaration must match the payload).
 """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if preprocessing is None:
        preprocessing = "standardise" if scaler is not None else "none"
    if scaler is not None and preprocessing != "standardise":
        raise ArtifactError("scaler given but preprocessing != 'standardise' — declare what the payload carries")
    if scaler is None and preprocessing == "standardise":
        raise ArtifactError("preprocessing='standardise' but no scaler in the payload")

    feature_version = str(feature_schema_config()["feature_schema_version"])
    fr14 = {
        "model_name": str(model_name),
        "model_version": str(model_version),
        "feature_version": feature_version,
        "training_dataset_version": str(training_dataset_version),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split_name": str(split_name),
        "seed": int(seed),
    }
    payload = {"fr14": fr14, "preprocessing": preprocessing, "features": list(features),
               "model": model, "scaler": scaler}
    if extra:
        payload.update(extra)
    joblib.dump(payload, path)

    digest = _sha256_of(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")

    try:
        reg = _registry()
        reg.register_model(
            model_name=fr14["model_name"],
            model_version=fr14["model_version"],
            feature_version=fr14["feature_version"],
            training_dataset_version=fr14["training_dataset_version"],
            artifact_path=str(path),
            provenance_hash=str(extra["provenance_hash"]) if extra and extra.get("provenance_hash") else None,
            split_name=fr14["split_name"],
            seed=fr14["seed"],
        )
    except ModelRegistryError as exc:
        # the sidecar + bundle exist; the registry refusal is surfaced, not swallowed
        raise ArtifactError(f"artifact saved to {path} but registry registration refused: {exc}") from exc
    return fr14


def verify_sidecar(path: str | Path) -> None:
    """Fail loudly unless the ``.sha256`` sidecar exists and matches the file."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ArtifactError(f"missing integrity sidecar {sidecar.name} — refusing to load an unverified artifact")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    got = _sha256_of(path)
    if got != expected:
        raise ArtifactError(
            f"artifact integrity failure: {path.name} sha256 {got[:12]}… != sidecar {expected[:12]}… "
            "(tampered or stale file)"
        )


def load_model_artifact(path: str | Path, *, expected_schema_version: str | None = None) -> dict[str, Any]:
    """Verify integrity, load, and assert the + preprocessing contract.

 ``expected_schema_version`` defaults to the live configs/feature_schema
 version — a model trained on another schema is refused, not silently
 applied. Returns ``{"model", "scaler", "features", "preprocessing",
 "fr14", "transform"}`` where ``transform(df)`` applies the declared
 preprocessing (or the identity for ``"none"``).
 """
    path = Path(path)
    verify_sidecar(path)
    payload = joblib.load(path)
    for field in ("fr14", "preprocessing", "features", "model"):
        if field not in payload:
            raise ArtifactError(f"artifact {path.name} lacks the required '{field}' block (pre-fixall bundle?)")
    fr14 = payload["fr14"]
    missing = [f for f in FR14_PAYLOAD_FIELDS if f not in fr14]
    if missing:
        raise ArtifactError(f"artifact {path.name}  block missing fields: {missing}")

    live_schema = str(feature_schema_config()["feature_schema_version"])
    want_schema = str(expected_schema_version) if expected_schema_version is not None else live_schema
    if str(fr14["feature_version"]) != want_schema:
        raise ArtifactError(
            f"artifact {path.name} was trained on feature schema {fr14['feature_version']!r} "
            f"but the live schema is {want_schema!r} — refusing to load"
        )

    preprocessing = payload["preprocessing"]
    scaler = payload.get("scaler")

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [c for c in payload["features"] if c not in df.columns]
        if missing_cols:
            raise ArtifactError(f"frame lacks artifact feature columns: {missing_cols}")
        if preprocessing == "standardise":
            return pd.DataFrame(scaler.transform(df[payload["features"]]), columns=payload["features"], index=df.index)
        return df[payload["features"]]

    return {
        "model": payload["model"],
        "scaler": scaler,
        "features": list(payload["features"]),
        "preprocessing": preprocessing,
        "fr14": fr14,
        "extras": {k: v for k, v in payload.items() if k not in {"fr14", "preprocessing", "features", "model", "scaler"}},
        "transform": transform,
    }
