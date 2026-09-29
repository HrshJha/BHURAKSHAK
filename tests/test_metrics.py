""" — evaluation metrics tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    HEADLINE_METRICS,
    MetricsError,
    brier_score,
    classification_metrics,
    expected_calibration_error,
    false_alarms_per_day,
    f1_score,
    hotspot_localisation_error,
    iou_hotspots,
    lead_time_stats,
    precision,
    pr_auc,
    recall,
    regression_metrics,
    reliability_curve,
)

# --- the accuracy exclusion (an acceptance check of its own) -----------------

def test_accuracy_is_not_a_headline_metric() -> None:
    assert "accuracy" not in HEADLINE_METRICS
    assert not any("accuracy" in name for name in HEADLINE_METRICS)


def test_metrics_module_exposes_no_accuracy_callables() -> None:
    import src.evaluation.metrics as m

    for name in dir(m):
        if "accuracy" in name.lower() and name != "observed_accuracy_reexport_guard":
            obj = getattr(m, name)
            assert not callable(obj), f"accuracy-like callable leaked: {name}"


# --- detection family --------------------------------------------------------

def test_precision_recall_f1_perfect_and_zero() -> None:
    y = [1, 1, 0, 0]
    assert precision(y, y) == 1.0
    assert recall(y, y) == 1.0
    assert f1_score(y, y) == 1.0
    assert precision(y, [0, 0, 0, 0]) == 0.0
    assert recall(y, [0, 0, 0, 0]) == 0.0
    assert f1_score(y, [0, 0, 0, 0]) == 0.0


def test_precision_zero_alerts_is_zero_not_error() -> None:
    assert precision([0, 0, 1], [0, 0, 0]) == 0.0


def test_confusion_counts_sum() -> None:
    from src.evaluation.metrics import confusion_counts

    c = confusion_counts([1, 0, 1, 0, 1], [1, 1, 0, 0, 1])
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    assert sum(c.values()) == 5


def test_pr_auc_matches_sklearn_on_clean_case() -> None:
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    s = rng.random(200)
    assert pr_auc(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-9)


def test_pr_auc_ties_absorbed_no_free_perfect_points() -> None:
    # all scores tied → ONE operating point; AP must be the pooled precision,
    # not a weighted average that rewards ordering the tie favourably.
    y = [1, 0, 1, 0]
    assert pr_auc(y, [0.7, 0.7, 0.7, 0.7]) == pytest.approx(0.5)


def test_pr_auc_perfect_and_inverted() -> None:
    assert pr_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)
    # inverted: positives surface at recall 0.5 (prec 1/3) and recall 1 (prec 1/2)
    assert pr_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(5.0 / 12.0)


def test_pr_auc_no_positives_raises() -> None:
    with pytest.raises(MetricsError, match="no positive"):
        pr_auc([0, 0, 0], [0.1, 0.2, 0.3])


def test_classification_metrics_bundle() -> None:
    out = classification_metrics([1, 1, 0, 0], [1, 0, 1, 0], y_score=[0.8, 0.4, 0.6, 0.2])
    assert out["tp"] == 1 and out["fp"] == 1 and out["fn"] == 1 and out["tn"] == 1
    # sweep: p=1@rec 0.5, then p=2/3@rec 1 → 0.5 + 0.5·2/3
    assert out["pr_auc"] == pytest.approx(5.0 / 6.0)
    assert "accuracy" not in out


# --- displacement-error family ----------------------------------------------

def test_regression_metrics_known_values() -> None:
    out = regression_metrics([1.0, 2.0, 3.0], [1.5, 2.0, 2.0])
    # errors: +0.5, 0, −1 → |err| mean 0.5, rmse sqrt((0.25+0+1)/3), bias −1/6
    assert out["mae"] == pytest.approx(0.5)
    assert out["rmse"] == pytest.approx(np.sqrt(1.25 / 3.0))
    assert out["max_abs_error"] == pytest.approx(1.0)
    assert out["bias"] == pytest.approx(-1.0 / 6.0)
    assert out["n_used"] == 3


def test_regression_metrics_drops_nan_pairs() -> None:
    out = regression_metrics([1.0, np.nan, 3.0], [1.5, 99.0, 3.0])
    assert out["n_used"] == 2
    assert out["max_abs_error"] == pytest.approx(0.5)


def test_regression_metrics_all_nan_raises() -> None:
    with pytest.raises(MetricsError, match="NaN"):
        regression_metrics([np.nan, np.nan], [1.0, 2.0])


# --- spatial family ----------------------------------------------------------

def test_iou_basic_and_conventions() -> None:
    assert iou_hotspots([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(1.0 / 3.0)
    assert iou_hotspots([1, 1], [1, 1]) == 1.0
    assert iou_hotspots([0, 0], [0, 0]) == 1.0, "both quiet maps agree perfectly"
    assert iou_hotspots([1, 0], [0, 0]) == 0.0


def test_hotspot_localisation_error_matched() -> None:
    true_xy = [(0.0, 0.0), (100.0, 0.0)]
    pred_xy = [(5.0, 0.0), (120.0, 0.0), (300.0, 300.0)]
    out = hotspot_localisation_error(true_xy, pred_xy, match_radius_m=50.0)
    assert out["mean_localisation_error_m"] == pytest.approx((5.0 + 20.0) / 2.0)
    assert out["matched_fraction"] == 1.0
    assert out["n_unmatched"] == 0


def test_hotspot_localisation_error_unmatched_capped_not_hidden() -> None:
    true_xy = [(0.0, 0.0), (500.0, 500.0)]
    pred_xy = [(1.0, 0.0)]
    out = hotspot_localisation_error(true_xy, pred_xy, match_radius_m=50.0)
    # the far hotspot misses; its distance is capped at the radius, not clipped away
    assert out["mean_localisation_error_m"] == pytest.approx((1.0 + 50.0) / 2.0)
    assert out["n_unmatched"] == 1
    assert out["matched_fraction"] == 0.5


def test_hotspot_localisation_error_no_predictions_is_inf() -> None:
    out = hotspot_localisation_error([(0.0, 0.0)], np.zeros((0, 2)), match_radius_m=25.0)
    assert out["mean_localisation_error_m"] == pytest.approx(25.0)  # capped infinity
    assert out["n_unmatched"] == 1
    assert out["matched_fraction"] == 0.0


# --- calibration family (re-exports) ----------------------------------------

def test_calibration_reexports_match_risk_module() -> None:
    import src.evaluation.metrics as m
    from src.risk.calibration import brier_score as bs_src
    from src.risk.calibration import expected_calibration_error as ece_src
    from src.risk.calibration import reliability_curve as rc_src

    assert m.brier_score is bs_src
    assert m.expected_calibration_error is ece_src
    assert m.reliability_curve is rc_src


def test_brier_and_ece_known_values() -> None:
    proba = np.array([[0.8, 0.2], [0.3, 0.7]])
    y = np.array([0, 1])
    # brier: (0.8-1)²+(0.2)²=0.08; (0.3)²+(0.7-1)²=0.18 → mean 0.13
    assert brier_score(proba, y) == pytest.approx(0.13)
    # perfectly confident-and-correct → ECE 0
    perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(perfect, y) == pytest.approx(0.0)


def test_reliability_curve_columns() -> None:
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]])
    y = np.array([0, 0, 1])
    curve = reliability_curve(proba, y, bins=10)
    assert {"bin_lower", "bin_upper", "count", "mean_confidence", "observed_accuracy"} <= set(curve.columns)
    assert int(curve["count"].sum()) == 3


# --- operational / temporal family ------------------------------------------

def test_false_alarms_per_day() -> None:
    # 2 false alarms on a 4-day span
    y_true = [0, 0, 0, 1, 0, 0]
    y_pred = [1, 0, 1, 1, 0, 0]
    assert false_alarms_per_day(y_true, y_pred, n_days=4.0) == pytest.approx(0.5)


def test_false_alarms_per_day_rejects_unstated_span() -> None:
    with pytest.raises(MetricsError, match="n_days"):
        false_alarms_per_day([0, 1], [1, 1], n_days=0.0)


def test_lead_time_stats_median_p10_and_missed_events() -> None:
    alerts = {"E1": 2.0, "E2": 0.0, "E3": 5.0}
    onsets = {"E1": 5.0, "E2": 4.0, "E3": 4.0}
    out = lead_time_stats(alerts, onsets, stride_hours=0.6)
    # leads: E1 = 3×0.6=1.8 h; E2 = 4×0.6=2.4 h; E3 alerted after onset → missed
    assert out["median_lead_time_hours"] == pytest.approx(2.1)
    assert out["p10_lead_time_hours"] == pytest.approx(np.percentile([1.8, 2.4], 10))
    assert out["n_events_with_lead"] == 2
    assert out["n_missed_events"] == 1
    assert out["missed_event_rate"] == pytest.approx(1.0 / 3.0)


def test_lead_time_stats_all_missed_is_nan_not_zero() -> None:
    out = lead_time_stats({}, {"E1": 3.0, "E2": 4.0}, stride_hours=0.6)
    assert np.isnan(out["median_lead_time_hours"])
    assert out["missed_event_rate"] == 1.0


def test_lead_time_stats_negative_lead_impossible() -> None:
    out = lead_time_stats({"E1": 1.0}, {"E1": 1.0}, stride_hours=0.6)
    assert out["median_lead_time_hours"] == pytest.approx(0.0)
    assert out["missed_event_rate"] == 0.0


def test_lead_time_stats_stride_must_be_positive() -> None:
    with pytest.raises(MetricsError, match="stride_hours"):
        lead_time_stats({"E1": 0.0}, {"E1": 1.0}, stride_hours=0.0)


# --- input guards ------------------------------------------------------------

@pytest.mark.parametrize("fn", [precision, recall, f1_score])
def test_length_mismatch_raises(fn) -> None:
    with pytest.raises(MetricsError):
        fn([1, 0], [1, 0, 0])


def test_non_binary_confusion_inputs_raise() -> None:
    with pytest.raises(MetricsError, match="binary"):
        precision([0.0, 2.0], [0.0, 1.0])
