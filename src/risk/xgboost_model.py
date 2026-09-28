"""XGBoost risk classifier — PRD §15, FR-6 (T-046).

§15: multimodal risk classification, the primary first serious risk model.

- Target: the 3-class MVP ``risk_label`` (NORMAL/WARNING/CRITICAL;
  GREEN/WATCH/WARNING/CRITICAL expansion comes after the 3-class model is
  validated — the classes are read from the data, not hard-coded).
- Inputs: engineered feature groups A–F (as available) + the Isolation Forest
  ``anomaly_score`` (T-043) + the ``physics_residual`` (T-040/T-041) — the §15
  "multimodal" contract.
- Output: per-class probabilities that sum to 1 (softmax objective), consumed
  by the §21.1 alert engine — the raw output never decides an action.

Hyperparameters from configs/risk_model.yaml (NFR-6); the caller supplies the
§23-safe split column (T-068) — this module never splits on its own and never
shuffles rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.config import risk_model_config

__all__ = ["XGBoostModelError", "RiskModel", "MODEL_INPUT_GROUPS", "train_risk_model"]

#: §15 input contract: feature groups A–F plus the two cross-model signals.
#: T-070 adds the Phase-4 modalities (G DGPS / H Sentinel-1) so the ablation
#: arms can switch them on; T-077 adds the §16 forecast features (I_forecast),
#: present only on frames joined via src/forecasting/forecast_to_risk.py —
#: the DEFAULT frame carries none of their columns, so behaviour for every
#: pre-existing caller is unchanged (absent ⇒ skipped).
MODEL_INPUT_GROUPS = (
    "A_physical",
    "B_temporal",
    "C_spatial",
    "D_vibration",
    "E_sensor_health",
    "F_physics",
    "G_dgps",
    "H_insar",
    "I_forecast",
)

#: Cross-model signals §15 adds on top of the feature groups.
EXTRA_SIGNALS = ("anomaly_score", "physics_residual")


class XGBoostModelError(ValueError):
    """Raised on invalid risk-model inputs or configuration."""


@dataclass
class RiskModel:
    model: XGBClassifier
    features: list[str]
    classes: list[str]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Per-class probabilities, columns ordered like ``classes``; rows sum to 1."""
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise XGBoostModelError(f"feature columns missing for inference: {missing}")
        proba = self.model.predict_proba(df[self.features].to_numpy(dtype=float))
        if proba.shape[1] != len(self.classes):
            raise XGBoostModelError(
                f"model emits {proba.shape[1]} class probabilities for {len(self.classes)} classes"
            )
        return proba

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.classes)[np.argmax(self.predict_proba(df), axis=1)]


def _resolve_features(df: pd.DataFrame) -> list[str]:
    """§15 inputs = union of available §13 group columns + the two signals."""
    schema_cols = set(df.columns)
    group_features: list[str] = []
    seen: set[str] = set()
    for group, names in (
        ("A_physical", ("tilt_x", "tilt_y", "tilt_magnitude", "displacement", "strain")),
        ("B_temporal", ("rolling_mean", "rolling_std", "rolling_min", "rolling_max", "slope",
                        "velocity", "acceleration", "trend", "persistence", "change_point_score")),
        ("C_spatial", ("neighbor_mean", "neighbor_std", "neighbor_anomaly_fraction", "spatial_coherence",
                       "local_gradient", "local_strain", "hotspot_density", "distance_to_subsidence_center")),
        ("D_vibration", ("vibration_rms", "vibration_peak", "crest_factor", "band_energy_low",
                         "band_energy_mid", "band_energy_high", "spectral_centroid")),
        ("E_sensor_health", ("battery", "RSSI", "SNR", "packet_loss", "missing_ratio",
                             "stuck_sensor_flag", "drift_score")),
        ("F_physics", ("expected_displacement", "expected_tilt", "physics_residual", "physics_residual_velocity")),
        # Phase-4 modalities (T-070 arms C/D/E): present only on frames joined
        # to DGPS control points / the mesh-aligned Sentinel-1 product.
        ("G_dgps", ("vertical_displacement", "horizontal_displacement", "velocity_dgps",
                    "acceleration_dgps", "mesh_vs_dgps_residual")),
        ("H_insar", ("LOS_displacement", "LOS_velocity", "LOS_acceleration", "cumulative_displacement",
                     "coherence", "spatial_gradient", "local_hotspot_density")),
        # §16 forecast features (T-077): ``forecast_{channel}_h{steps}`` —
        # physical-unit forecasts entering the risk layer as FEATURES; the
        # names are pattern-matched so any horizon config (T-078) resolves.
        ("I_forecast", tuple(sorted(n for n in schema_cols if n.startswith("forecast_")))),
    ):
        assert group in MODEL_INPUT_GROUPS
        for name in names:
            if name in schema_cols and name not in seen:
                group_features.append(name)
                seen.add(name)
    for signal in EXTRA_SIGNALS:
        if signal in schema_cols and signal not in seen:
            group_features.append(signal)
            seen.add(signal)
    if not group_features:
        raise XGBoostModelError("no §15 model inputs found in the frame")
    return group_features


def train_risk_model(df: pd.DataFrame, feature_groups: list[str] | None = None) -> RiskModel:
    """Fit the §15 classifier on the TRAIN split of ``df``.

    ``df`` must carry a ``split`` column (T-068) and the 3-class ``risk_label``.
    Hyperparameters come from configs/risk_model.yaml; the objective is
    multi-class softmax so per-class probabilities sum to 1 (§15).
    """
    cfg = risk_model_config()["xgboost"]
    for col in ("split", "risk_label"):
        if col not in df.columns:
            raise XGBoostModelError(f"frame must carry {col!r}")

    features = _resolve_features(df)
    if feature_groups is not None:
        wanted = {n for n in feature_groups}
        features = [f for f in features if f in wanted] or features

    train = df[df["split"] == "train"]
    val = df[df["split"] == "validation"]
    if train.empty or val.empty:
        raise XGBoostModelError("both 'train' and 'validation' splits must be non-empty")

    classes = sorted(train["risk_label"].unique())
    if len(classes) != 3:
        raise XGBoostModelError(
            f"§15 MVP expects exactly 3 classes, found {classes} — expand after validation"
        )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = train["risk_label"].map(class_to_idx).to_numpy()

    model = XGBClassifier(
        n_estimators=int(cfg["n_estimators"]),
        max_depth=int(cfg["max_depth"]),
        learning_rate=float(cfg["learning_rate"]),
        subsample=float(cfg["subsample"]),
        colsample_bytree=float(cfg["colsample_bytree"]),
        reg_lambda=float(cfg["reg_lambda"]),
        random_state=int(cfg["random_state"]),
        objective="multi:softmax",
        num_class=len(classes),
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        n_jobs=-1,
    )
    model.fit(
        train[features].to_numpy(dtype=float),
        y,
        eval_set=[(val[features].to_numpy(dtype=float),
                   val["risk_label"].map(class_to_idx).to_numpy())],
        verbose=False,
    )
    return RiskModel(model=model, features=features, classes=[str(c) for c in classes])
