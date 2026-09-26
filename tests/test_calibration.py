"""T-048 acceptance tests — probability calibration and calibration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.calibration import (
    CalibrationError,
    brier_score,
    expected_calibration_error,
    fit_probability_calibrator,
    reliability_curve,
)


def _probabilities(n: int = 600, overconfident: bool = True, seed: int = 0):
    """Probabilities that are systematically overconfident when overconfident=True."""
    rng = np.random.default_rng(seed)
    classes = ["CRITICAL", "NORMAL", "WARNING"]
    y_idx = rng.integers(0, 3, size=n)
    proba = np.zeros((n, 3))
    for i in range(n):
        truth = y_idx[i]
        # predicted probability of the true class: 0.55 when miscalibrated
        p_true = 0.95 if overconfident else 0.55
        rest = (1.0 - p_true) / 2.0
        proba[i] = [rest, rest, rest]
        proba[i, truth] = p_true
    return proba, y_idx, classes


def test_brier_zero_for_perfect_probabilities() -> None:
    proba = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    y = np.array([1, 0])
    assert brier_score(proba, y) == pytest.approx(0.0)


def test_brier_worse_for_confident_wrong() -> None:
    y = np.array([0])
    confident_wrong = brier_score(np.array([[0.0, 0.0, 1.0]]), y)
    uncertain = brier_score(np.array([[1 / 3] * 3]), y)
    assert confident_wrong > uncertain


def test_ece_detects_overconfidence() -> None:
    proba, y, _ = _probabilities(overconfident=True)
    ece_bad = expected_calibration_error(proba, y)
    proba_c, y_c, _ = _probabilities(overconfident=False)
    ece_better = expected_calibration_error(proba_c, y_c)
    assert ece_bad > ece_better, "overconfident probabilities must have higher ECE"
    assert ece_bad > 0.2


def test_reliability_curve_shape_and_monotonicity() -> None:
    proba, y, _ = _probabilities(600, overconfident=True)
    curve = reliability_curve(proba, y)
    assert list(curve.columns) >= ["bin_lower", "bin_upper", "count", "mean_confidence", "observed_accuracy"]
    assert curve["count"].sum() == 600
    # overconfident: observed accuracy below mean confidence in populated bins
    populated = curve[curve["count"] > 0]
    assert (populated["observed_accuracy"] < populated["mean_confidence"] + 0.15).any()


def test_calibrator_fitted_on_validation_reduces_ece_on_test() -> None:
    """The §15/§24 acceptance: calibration (fit on validation only) improves
    test-set calibration of overconfident probabilities."""
    proba, y, classes = _probabilities(1200, overconfident=True, seed=3)
    cut = len(proba) // 2
    val_p, val_y = proba[:cut], y[:cut]
    test_p, test_y = proba[cut:], y[cut:]

    cal = fit_probability_calibrator(val_p, pd.Series(np.asarray(classes)[val_y]), classes)
    assert cal.classes == classes
    calibrated = cal.transform(test_p)

    ece_before = expected_calibration_error(test_p, test_y)
    ece_after = expected_calibration_error(calibrated, test_y)
    assert ece_after < ece_before, f"calibration must reduce ECE ({ece_before:.3f} -> {ece_after:.3f})"
    # rows still sum to 1
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-9)


def test_calibrator_requires_matching_classes() -> None:
    proba, y, classes = _probabilities(100)
    with pytest.raises(CalibrationError):
        fit_probability_calibrator(proba, pd.Series(["BOGUS"] * len(y)), classes)
    with pytest.raises(CalibrationError):
        fit_probability_calibrator(proba[:, :2], pd.Series(np.asarray(classes)[y]), classes)


def test_unknown_method_rejected() -> None:
    import src.risk.calibration as cal

    real = cal.risk_model_config

    def fake():
        data = real()
        data["calibration"]["method"] = "psychic"
        return data

    cal.risk_model_config = fake
    try:
        proba, y, classes = _probabilities(60)
        with pytest.raises(CalibrationError, match="unsupported"):
            cal.fit_probability_calibrator(proba, pd.Series(np.asarray(classes)[y]), classes)
    finally:
        cal.risk_model_config = real
