"""Probability calibration — PRD §15 (calibrated probabilities), §24 (T-048).

§24: Brier score, a reliability curve and expected calibration error are the
calibration metrics; calibration is fitted on a validation split **disjoint
from training** (§15: "calibrated class probabilities" are the model output).

Method: **confidence calibration** — an isotonic regression maps the row's
maximum raw probability (the model's confidence) to the observed accuracy at
that confidence level on the validation split. At inference the top class's
probability is replaced by the mapped value and the remainder is redistributed
proportionally over the other classes, so rows still sum to exactly 1 (§15).
Calibrating each class independently and renormalising would re-inflate the
top class after the sum constraint — the known multiclass distortion this
design avoids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import risk_model_config

__all__ = ["CalibrationError", "ProbabilityCalibrator", "brier_score", "expected_calibration_error", "reliability_curve"]

_BINS = 10


class CalibrationError(ValueError):
    """Raised on invalid calibration inputs."""


@dataclass
class ProbabilityCalibrator:
    """Confidence calibrator: isotonic mapping fitted on validation data only."""

    edges: np.ndarray  # (_BINS+1,) confidence bin edges
    values: np.ndarray  # (_BINS,) calibrated confidence per bin
    classes: list[str]

    def _calibrated_confidence(self, confidence: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(self.edges, confidence, side="right") - 1, 0, _BINS - 1)
        return self.values[idx]

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Map raw probabilities through the fitted confidence calibration.

        The top class receives the calibrated confidence; the other classes
        share the remainder proportionally. Rows sum to exactly 1 (§15).
        """
        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2 or proba.shape[1] != len(self.classes):
            raise CalibrationError(f"expected (n, {len(self.classes)}) probabilities")
        confidence = proba.max(axis=1)
        top = proba.argmax(axis=1)
        cal_conf = self._calibrated_confidence(confidence)

        out = proba.copy()
        n_classes = proba.shape[1]
        for i in range(proba.shape[0]):
            others = [j for j in range(n_classes) if j != top[i]]
            rest_sum = float(out[i, others].sum())
            out[i, top[i]] = cal_conf[i]
            if rest_sum > 0:
                out[i, others] *= (1.0 - cal_conf[i]) / rest_sum
            else:
                out[i, others] = (1.0 - cal_conf[i]) / (n_classes - 1)
        return out


def fit_probability_calibrator(
    raw_proba: np.ndarray,
    labels: pd.Series,
    classes: list[str],
) -> ProbabilityCalibrator:
    """Fit the calibrator on VALIDATION rows only (disjoint from training).

    ``labels`` must be the true ``risk_label`` values aligned with ``raw_proba``.
    """
    cfg = risk_model_config()["calibration"]
    if cfg["method"] != "isotonic":
        raise CalibrationError(f"unsupported calibration method {cfg['method']!r}")
    raw_proba = np.asarray(raw_proba, dtype=float)
    if raw_proba.ndim != 2 or raw_proba.shape[1] != len(classes):
        raise CalibrationError("proba shape must be (n_samples, n_classes)")
    if len(labels) != len(raw_proba):
        raise CalibrationError("labels must align with probabilities")

    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = labels.map(label_to_idx).to_numpy()
    if np.isnan(y_idx.astype(float)).any():
        raise CalibrationError("labels contain classes outside the model's classes")

    confidence = raw_proba.max(axis=1)
    correct = (raw_proba.argmax(axis=1) == y_idx).astype(float)

    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(confidence, correct)
    edges = np.linspace(0.0, 1.0, _BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    values = np.clip(iso.predict(centers), 0.0, 1.0)
    return ProbabilityCalibrator(edges=edges, values=values, classes=list(classes))


def brier_score(proba: np.ndarray, y_true_idx: np.ndarray) -> float:
    """Multiclass Brier score: mean over samples of Σ_k (p_k − 1{y=k})²."""
    proba = np.asarray(proba, dtype=float)
    n, k = proba.shape
    onehot = np.zeros_like(proba)
    onehot[np.arange(n), np.asarray(y_true_idx, dtype=int)] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def expected_calibration_error(proba: np.ndarray, y_true_idx: np.ndarray, bins: int = _BINS) -> float:
    """ECE: Σ_bins (n_b / N) · |accuracy_b − confidence_b|."""
    proba = np.asarray(proba, dtype=float)
    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == np.asarray(y_true_idx)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    n = len(confidence)
    ece = 0.0
    for i in range(bins):
        mask = (confidence > edges[i]) & (confidence <= edges[i + 1]) if i else (confidence <= edges[1])
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def reliability_curve(proba: np.ndarray, y_true_idx: np.ndarray, bins: int = _BINS) -> pd.DataFrame:
    """Per-bin (confidence, observed accuracy, count) for the reliability diagram."""
    proba = np.asarray(proba, dtype=float)
    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == np.asarray(y_true_idx)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        mask = (confidence > edges[i]) & (confidence <= edges[i + 1]) if i else (confidence <= edges[1])
        rows.append(
            {
                "bin_lower": edges[i],
                "bin_upper": edges[i + 1],
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
                "observed_accuracy": float(correct[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)
