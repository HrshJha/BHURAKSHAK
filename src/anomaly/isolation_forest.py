"""Isolation Forest module — PRD §14, FR-5 (T-043).

§14: the detector answers "does this look abnormal?" — never "is this
subsidence?". It is trained **only on healthy-baseline windows**, unsupervised,
and its output feeds forward as ONE input feature to XGBoost
(``anomaly_score = -model.score_samples(X)``), never interpreted directly as a
subsidence probability.

Hyperparameters come from configs/anomaly.yaml and replicate the §14 code
block exactly: ``IsolationForest(n_estimators=300, contamination="auto",
random_state=42)``.

Healthy-baseline rule (Gap G-2, resolved for the 3-class MVP): §14's literal
``df[df["risk_label"] == "GREEN"]`` is inoperable because the MVP vocabulary
has no GREEN. The healthy mask here is the triple-healthy conjunction

    anomaly_label == 0  AND  fault_label == "NONE"  AND  risk_label == "NORMAL"

so sensor-faulted windows (which carry ``risk_label == NORMAL`` under a fault)
are excluded from the baseline — keeping §12's "sensor is broken ≠ ground is
moving" separation true in the training set itself.

Split discipline: the module accepts an explicit ``split`` column (produced by
the caller from the §23 event/parameter-regime logic); the decision threshold
is the ``far_alpha``-quantile of **validation** healthy scores, so the
false-alarm rate on unseen healthy data ≈ 1 − far_alpha by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import anomaly_config

__all__ = [
    "IsolationForestError",
    "IsolationForestModel",
    "healthy_baseline_mask",
    "train_isolation_forest",
]

#: The three §14 ablation feature groups, in order (A physical, B +temporal, C +spatial).
ABLATION_STEPS = ("A_physical", "B_add_temporal", "C_add_spatial")


class IsolationForestError(ValueError):
    """Raised on invalid Isolation-Forest inputs or configuration."""


@dataclass
class IsolationForestModel:
    """Fitted detector + the metadata needed to score honestly."""

    model: IsolationForest
    features: list[str]
    threshold: float
    threshold_rule: str
    n_training_windows: int

    def anomaly_score(self, df: pd.DataFrame) -> np.ndarray:
        """§14: anomaly_score = -score_samples(X). Higher = more anomalous."""
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise IsolationForestError(f"feature columns missing for scoring: {missing}")
        return -self.model.score_samples(df[self.features].to_numpy(dtype=float))

    def flags(self, df: pd.DataFrame) -> np.ndarray:
        """Boolean anomaly flags at the fitted threshold."""
        return self.anomaly_score(df) > self.threshold


def healthy_baseline_mask(df: pd.DataFrame) -> np.ndarray:
    """The G-2 triple-healthy mask (see module docstring).

    ``risk_label == NORMAL`` is required only when the column exists; tables
    without labels (pure inference frames) cannot assert healthiness and raise.
    """
    for col in ("anomaly_label", "fault_label"):
        if col not in df.columns:
            raise IsolationForestError(f"healthy-baseline mask needs the §12 column {col!r}")
    mask = (pd.to_numeric(df["anomaly_label"], errors="coerce") == 0).to_numpy()
    mask &= (df["fault_label"] == "NONE").to_numpy()
    if "risk_label" in df.columns:
        mask &= (df["risk_label"] == "NORMAL").to_numpy()
    return mask


def _resolve_features(groups: list[str] | None) -> list[str]:
    cfg = anomaly_config()["ablation"]
    if groups is None:
        groups = list(ABLATION_STEPS)
    unknown = [g for g in groups if g not in cfg]
    if unknown:
        raise IsolationForestError(f"unknown ablation feature groups: {unknown}")
    features: list[str] = []
    for g in groups:
        for name in cfg[g]:
            if name not in features:
                features.append(str(name))
    return features


def train_isolation_forest(
    df: pd.DataFrame,
    feature_groups: list[str] | None = None,
) -> IsolationForestModel:
    """Fit the §14 detector on healthy-baseline windows of the TRAIN split only.

    ``df`` must carry a ``split`` column with at least the values ``train`` and
    ``validation``. The threshold is set so that ``far_alpha`` (configs/
    anomaly.yaml, default 0.99) of VALIDATION healthy windows stay below it.
    """
    cfg = anomaly_config()["isolation_forest"]
    for col in ("split",):
        if col not in df.columns:
            raise IsolationForestError("frame must carry a 'split' column (§23 discipline)")

    features = _resolve_features(feature_groups)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise IsolationForestError(f"feature columns missing: {missing}")

    train = df[df["split"] == "train"]
    val = df[df["split"] == "validation"]
    if train.empty or val.empty:
        raise IsolationForestError("both 'train' and 'validation' splits must be non-empty")

    train_healthy = train[healthy_baseline_mask(train)]
    if len(train_healthy) < 100:
        raise IsolationForestError(
            f"healthy training baseline too small ({len(train_healthy)} rows) for §14 training"
        )

    model = IsolationForest(
        n_estimators=int(cfg["n_estimators"]),
        contamination=cfg["contamination"],
        random_state=int(cfg["random_state"]),
        n_jobs=-1,
    )
    model.fit(train_healthy[features].to_numpy(dtype=float))

    far_alpha = float(anomaly_config()["far_alpha"])
    val_healthy_scores = -model.score_samples(val[healthy_baseline_mask(val)][features].to_numpy(dtype=float))
    if val_healthy_scores.size < 10:
        raise IsolationForestError("too few validation healthy windows to set a threshold")
    threshold = float(np.quantile(val_healthy_scores, far_alpha))

    return IsolationForestModel(
        model=model,
        features=features,
        threshold=threshold,
        threshold_rule=f"p{far_alpha:.2f} of validation healthy scores (false-alarm ≈ {1.0 - far_alpha:.2f})",
        n_training_windows=int(len(train_healthy)),
    )
