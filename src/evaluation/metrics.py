"""Evaluation metrics —.

The metric surface for every study, ablation and notebook in the repo.
Five families:

- **Detection** (alert quality at the operating point): precision, recall,
 F1, and PR-AUC over the full score sweep. A confusion cell count rides
 along so no number is unexplained.

- **Displacement error** ( names it explicitly): MAE, RMSE,
 max-abs-error and signed bias of predicted vs reference displacement.

- **Spatial**: hotspot IoU (overlap of flagged node sets) and hotspot
 localisation error (how far the nearest predicted hotspot sits from each
 true one).

- **Calibration**: Brier score, reliability curve and expected calibration
 error — re-exported from:mod:`src.risk.calibration` so has
 ONE implementation, not two drifting copies.

- **Operational / temporal**: false alarms per day, median and P10 lead
 time, missed-event rate.

**Accuracy is deliberately absent.** replaces accuracy with the
class-balanced and cost-aware families above; in a subsidence mesh the
healthy class dominates so accuracy flatters a model that never warns. This
module exposes no ``accuracy`` function and the headline registry
(:data:`HEADLINE_METRICS`) omits it — asserted by tests/test_metrics.py.
"""

from __future__ import annotations

import numpy as np

from src.risk.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)

__all__ = [
    "MetricsError",
    "HEADLINE_METRICS",
    "precision",
    "recall",
    "f1_score",
    "pr_auc",
    "confusion_counts",
    "classification_metrics",
    "regression_metrics",
    "iou_hotspots",
    "hotspot_localisation_error",
    "false_alarms_per_day",
    "lead_time_stats",
    # calibration family (single implementation lives in src/risk/calibration.py)
    "brier_score",
    "expected_calibration_error",
    "reliability_curve",
]

#: The headline registry. `accuracy` is intentionally NOT a member —
#: see the module docstring. tests/test_metrics.py enforces this.
HEADLINE_METRICS = (
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "mae",
    "rmse",
    "max_abs_error",
    "bias",
    "iou_hotspots",
    "hotspot_localisation_error_m",
    "brier",
    "ece",
    "false_alarms_per_day",
    "median_lead_time_hours",
    "p10_lead_time_hours",
    "missed_event_rate",
)


class MetricsError(ValueError):
    """Raised on invalid metric inputs (mismatched lengths, empty data…)."""


def _paired(a, b, name: str) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise MetricsError(f"{name}: shapes differ {a.shape} vs {b.shape}")
    if a.size == 0:
        raise MetricsError(f"{name}: nothing to score")
    return a, b


# Detection family

def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """Binary confusion counts (1 = alert/event, 0 = quiet)."""
    t, p = _paired(y_true, y_pred, "confusion_counts")
    if not np.isin(t, (0.0, 1.0)).all() or not np.isin(p, (0.0, 1.0)).all():
        raise MetricsError("confusion_counts expects binary 0/1 inputs")
    t_i, p_i = t.astype(int), p.astype(int)
    return {
        "tp": int(((t_i == 1) & (p_i == 1)).sum()),
        "fp": int(((t_i == 0) & (p_i == 1)).sum()),
        "fn": int(((t_i == 1) & (p_i == 0)).sum()),
        "tn": int(((t_i == 0) & (p_i == 0)).sum()),
    }


def precision(y_true, y_pred) -> float:
    """TP / (TP + FP); 0.0 when the model raises no alert (never undefined)."""
    c = confusion_counts(y_true, y_pred)
    denom = c["tp"] + c["fp"]
    return c["tp"] / denom if denom else 0.0


def recall(y_true, y_pred) -> float:
    """TP / (TP + FN); 0.0 when there is no event to catch."""
    c = confusion_counts(y_true, y_pred)
    denom = c["tp"] + c["fn"]
    return c["tp"] / denom if denom else 0.0


def f1_score(y_true, y_pred) -> float:
    """Harmonic mean of precision and recall (0.0 when both are 0)."""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2.0 * p * r / (p + r) if (p + r) else 0.0


def pr_auc(y_true, y_score) -> float:
    """Area under the precision-recall curve (step-wise, tie-safe).

 Scores are swept from highest to lowest; equal scores are absorbed into
 one operating point (precision is read AFTER the whole tie group, which
 is the honest convention — a tie can never grant a free perfect point).
 """
    y, s = _paired(y_true, y_score, "pr_auc")
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order] > 0
    s_sorted = s[order]

    tp = fp = 0
    total_pos = int(y_sorted.sum())
    if total_pos == 0:
        raise MetricsError("pr_auc undefined with no positive examples")
    auc = 0.0
    prev_recall = 0.0
    i = 0
    n = len(y_sorted)
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:  # absorb the whole tie group
            tp += int(y_sorted[j])
            fp += int(not y_sorted[j])
            j += 1
        rec = tp / total_pos
        prec = tp / (tp + fp)
        auc += (rec - prev_recall) * prec
        prev_recall = rec
        i = j
    return float(auc)


def classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    """The detection dict: precision/recall/F1 (+ PR-AUC when scores exist)."""
    out = {
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        **confusion_counts(y_true, y_pred),
    }
    if y_score is not None:
        out["pr_auc"] = pr_auc(y_true, y_score)
    return out


# Displacement-error family

def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, max-abs-error and signed bias of displacement errors.

 NaN pairs are dropped (reported as ``n_used``); if nothing remains the
 call raises — an empty agreement claim is not an agreement claim.
 """
    t, p = _paired(y_true, y_pred, "regression_metrics")
    ok = np.isfinite(t) & np.isfinite(p)
    n_used = int(ok.sum())
    if n_used == 0:
        raise MetricsError("regression_metrics: all pairs are NaN — nothing to score")
    err = p[ok] - t[ok]
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(np.abs(err).max()),
        "bias": float(err.mean()),
        "n_used": n_used,
    }


# Spatial family

def iou_hotspots(pred_mask, true_mask) -> float:
    """IoU of flagged hotspot node sets.

 Convention: both empty → 1.0 (the maps agree perfectly that nothing
 burns); exactly one empty → 0.0.
 """
    p, t = _paired(pred_mask, true_mask, "iou_hotspots")
    if not np.isin(p, (0.0, 1.0)).all() or not np.isin(t, (0.0, 1.0)).all():
        raise MetricsError("iou_hotspots expects binary 0/1 masks")
    p_set = p > 0
    t_set = t > 0
    union = int((p_set | t_set).sum())
    if union == 0:
        return 1.0
    inter = int((p_set & t_set).sum())
    return inter / union


def hotspot_localisation_error(
    true_xy,
    pred_xy,
    *,
    match_radius_m: float,
) -> dict[str, float]:
    """How well predicted hotspot centres cover the TRUE ones (metres).

 For each true hotspot: the distance to the nearest predicted centre.
 ``match_radius_m`` decides whether that true hotspot counts as matched
 (a miss beyond the radius is still REPORTED at its capped distance, so
 a totally absent prediction cannot hide behind a clip). Returns the
 mean/max distance and the unmatched count.
 """
    if match_radius_m <= 0:
        raise MetricsError("match_radius_m must be positive")
    true_arr = np.asarray(true_xy, dtype=float)
    pred_arr = np.asarray(pred_xy, dtype=float)
    if true_arr.ndim != 2 or true_arr.shape[1] != 2:
        raise MetricsError("true_xy must be (n_true, 2)")
    if pred_arr.ndim != 2 or pred_arr.shape[1] != 2:
        raise MetricsError("pred_xy must be (n_pred, 2)")
    if len(true_arr) == 0:
        raise MetricsError("hotspot_localisation_error: no true hotspots given")

    if len(pred_arr) == 0:
        dists = np.full(len(true_arr), np.inf)
    else:
        d = np.sqrt(((true_arr[:, None, :] - pred_arr[None, :, :]) ** 2).sum(-1))
        dists = d.min(axis=1)

    capped = np.minimum(dists, float(match_radius_m))
    return {
        "mean_localisation_error_m": float(capped.mean()),
        "max_localisation_error_m": float(capped.max()),
        "n_true_hotspots": int(len(true_arr)),
        "n_unmatched": int((dists > match_radius_m).sum()),
        "matched_fraction": float((dists <= match_radius_m).mean()),
    }


# Operational / temporal family

def false_alarms_per_day(y_true, y_pred, *, n_days: float) -> float:
    """False alerts on quiet windows, per day of the observation span.

 ``n_days`` is STATED by the caller (the honest span the alerts were
 collected over) — never inferred from window counts.
 """
    c = confusion_counts(y_true, y_pred)
    if n_days <= 0:
        raise MetricsError("n_days must be positive — state the observation span")
    return c["fp"] / float(n_days)


def lead_time_stats(
    alert_window_by_event: dict[str, float],
    onset_window_by_event: dict[str, float],
    *,
    stride_hours: float,
) -> dict[str, float]:
    """Median / P10 lead time and missed-event rate across events.

 ``alert_window_by_event``: first alert window index per event (absent ⇒
 the event was never alerted). ``onset_window_by_event``: the ground-truth
 onset window index per event (REQUIRED for every evaluated event).
 Lead time = (onset − first alert) × ``stride_hours`` for events whose
 first alert is at or before onset; alert-after-onset and never-alerted
 events count as MISSED and enter no percentile.
 """
    if stride_hours <= 0:
        raise MetricsError("stride_hours must be positive")
    events = set(onset_window_by_event)
    alerts = set(alert_window_by_event) & events
    if not events:
        raise MetricsError("lead_time_stats: no events to score")

    leads: list[float] = []
    missed = 0
    for event in sorted(events):
        onset = float(onset_window_by_event[event])
        if event in alerts:
            alert = float(alert_window_by_event[event])
            if alert <= onset:
                leads.append((onset - alert) * stride_hours)
                continue
        missed += 1

    n = len(events)
    if leads:
        arr = np.asarray(leads, dtype=float)
        return {
            "median_lead_time_hours": float(np.median(arr)),
            "p10_lead_time_hours": float(np.percentile(arr, 10)),
            "n_events_with_lead": len(arr),
            "n_missed_events": missed,
            "missed_event_rate": missed / n,
        }
    return {
        "median_lead_time_hours": float("nan"),
        "p10_lead_time_hours": float("nan"),
        "n_events_with_lead": 0,
        "n_missed_events": missed,
        "missed_event_rate": missed / n,
    }
