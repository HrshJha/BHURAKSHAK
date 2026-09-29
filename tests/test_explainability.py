""" tests — / explainability contribution breakdown."""

from __future__ import annotations

import pytest

from src.risk.alert_engine import probability_of
from src.risk.explainability import (
    ExplanationError,
    RiskExplanation,
    SIGNAL_CATALOGUE,
    emit_risk_output,
)

FULL_SIGNALS = {
    "tilt_velocity": 0.21,
    "neighbour_anomaly_count": 5,
    "displacement_trend": 1.0,
    "insar_agreement": 1.0,
    "physics_residual": 0.2,  # normalised z-score: low → physically consistent
}


def test_every_risk_output_carries_a_breakdown() -> None:
    level, proba, expl = emit_risk_output(
        "n3",
        {"NORMAL": 0.05, "WARNING": 0.35, "CRITICAL": 0.60},
        FULL_SIGNALS,
    )
    assert level == "CRITICAL"
    assert len(expl.contributions) == len(FULL_SIGNALS)
    assert expl.predicted_level == level and expl.probabilities is proba
    lines = expl.summary
    assert len(lines) == len(FULL_SIGNALS)


def test_emitting_a_bare_probability_without_breakdown_raises() -> None:
    with pytest.raises(ExplanationError, match=""):
        emit_risk_output("n3", {"NORMAL": 0.9, "WARNING": 0.1, "CRITICAL": 0.0}, {})


def test_probability_validation_still_applies() -> None:
    with pytest.raises(ExplanationError, match="missing model classes"):
        emit_risk_output("n3", {"NORMAL": 1.0}, FULL_SIGNALS)
    with pytest.raises(ExplanationError, match="sum to 1"):
        emit_risk_output("n3", {"NORMAL": 0.5, "WARNING": 0.5, "CRITICAL": 0.5}, FULL_SIGNALS)


def test_contributions_follow_the_signal_catalogue() -> None:
    _, _, expl = emit_risk_output("n3", {"NORMAL": 0.05, "WARNING": 0.35, "CRITICAL": 0.60}, FULL_SIGNALS)
    by_signal = {c.signal: c for c in expl.contributions}
    assert by_signal["tilt_velocity"].raises_risk is True
    assert "high tilt velocity" in by_signal["tilt_velocity"].text
    assert by_signal["neighbour_anomaly_count"].text == "5 neighbouring nodes anomalous"
    # physics residual z=+0.2 is near zero → "physically consistent" → raises confidence
    assert by_signal["physics_residual"].raises_risk is True
    assert "physically consistent" in by_signal["physics_residual"].text


def test_physics_residual_counts_against_when_large() -> None:
    signals = dict(FULL_SIGNALS, physics_residual=4.2)  # 4.2σ — physics disagrees
    _, _, expl = emit_risk_output("n3", {"NORMAL": 0.05, "WARNING": 0.35, "CRITICAL": 0.60}, signals)
    phys = next(c for c in expl.contributions if c.signal == "physics_residual")
    assert phys.raises_risk is False
    assert "physics disagrees" in phys.text


def test_negative_values_produce_reassuring_phrases() -> None:
    signals = {"tilt_velocity": -0.1, "displacement_trend": -1.0, "physics_residual": 0.1}
    _, _, expl = emit_risk_output("n3", {"NORMAL": 0.9, "WARNING": 0.1, "CRITICAL": 0.0}, signals)
    by_signal = {c.signal: c for c in expl.contributions}
    assert by_signal["tilt_velocity"].raises_risk is False
    assert by_signal["tilt_velocity"].text == "tilt velocity quiet"
    assert by_signal["physics_residual"].raises_risk is True


def test_unknown_signals_are_included_honestly_as_neutral() -> None:
    _, _, expl = emit_risk_output("n3", {"NORMAL": 0.9, "WARNING": 0.1, "CRITICAL": 0.0}, {"new_sensor_metric": 2.0})
    (c,) = expl.contributions
    assert c.raises_risk is None, "no direction claimed for a signal outside the catalogue"
    assert c.text == "signal reported: new_sensor_metric"


def test_summary_signs_and_render_block() -> None:
    level, proba, expl = emit_risk_output(
        "n3",
        {"NORMAL": 0.05, "WARNING": 0.35, "CRITICAL": 0.60},
        {"tilt_velocity": 0.21, "physics_residual": 4.2},
    )
    lines = {c.signal: line for c, line in zip(expl.contributions, expl.summary)}
    assert lines["tilt_velocity"].startswith("+ "), "raising signals get '+'"
    assert lines["physics_residual"].startswith("- "), "counting-against signals get '−'"
    assert "tilt_velocity=0.21" in lines["tilt_velocity"]
    block = expl.render()
    assert block.startswith("Risk: CRITICAL (P=0.60)")
    assert "+ high tilt velocity" in block
    # contract: the emitted bundle always carries level + probability + breakdown
    assert (level, proba["CRITICAL"], bool(expl.summary)) == ("CRITICAL", 0.60, True)


def test_ties_resolve_to_the_more_severe_class_deterministically() -> None:
    level, _, _ = emit_risk_output(
        "n3",
        {"NORMAL": 0.0, "WARNING": 0.5, "CRITICAL": 0.5},
        FULL_SIGNALS,
    )
    assert level == "CRITICAL"


def test_argmax_matches_plain_numpy_argmax_on_ordinary_inputs() -> None:
    proba = {"NORMAL": 0.1, "WARNING": 0.7, "CRITICAL": 0.2}
    level, values, _ = emit_risk_output("n3", proba, FULL_SIGNALS)
    assert level == "WARNING"
    assert values == {"NORMAL": 0.1, "WARNING": 0.7, "CRITICAL": 0.2}


def test_catalogue_covers_documented_examples() -> None:
    for expected in ("tilt_velocity", "neighbour_anomaly_count", "displacement_trend", "insar_agreement", "physics_residual"):
        assert expected in SIGNAL_CATALOGUE


def test_dashboard_can_pair_aggregate_probability_with_the_breakdown() -> None:
    """P(WATCH or higher) from composes with the breakdown."""
    proba = {"NORMAL": 0.2, "WARNING": 0.5, "CRITICAL": 0.3}
    level, _, expl = emit_risk_output("n3", proba, FULL_SIGNALS)
    watch_or_higher = probability_of("WATCH_or_higher", proba)
    assert watch_or_higher == pytest.approx(0.8)
    assert expl.render().startswith(f"Risk: {level}")
