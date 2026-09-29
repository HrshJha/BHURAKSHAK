""" tests — alert-engine state machine."""

from __future__ import annotations

import pytest

from src.config import alerts_config, escalation_thresholds
from src.risk.alert_engine import (
    AlertEngine,
    AlertEngineError,
    probability_of,
)

#: escalation table as configured (mirrors configs/alerts.yaml MVP values).
ESC = {
    "GREEN_to_WATCH": {"probability_of": "WATCH_or_higher", "class_threshold": 0.5, "persistence_windows": 3, "requires": []},
    "WATCH_to_WARNING": {
        "probability_of": "WARNING_or_higher",
        "class_threshold": 0.6,
        "persistence_windows": 3,
        "requires": ["spatial_coherence_above_threshold", "displacement_trend_positive"],
    },
    "WARNING_to_CRITICAL": {
        "probability_of": "CRITICAL",
        "class_threshold": 0.7,
        "persistence_windows": 2,
        "requires": ["physics_residual_low", "neighbour_confirmation"],
        "min_confirming_neighbours": 2,
    },
}

CONFIRMED = {"physics_residual_low": True, "neighbour_confirmations": 3}

#: every requirement evidenced at once (the fastest legal escalation path)
FULLY_CONFIRMED = {
    "spatial_coherence_above_threshold": True,
    "displacement_trend_positive": True,
    "physics_residual_low": True,
    "neighbour_confirmations": 3,
}


def probs(normal: float, warning: float, critical: float) -> dict[str, float]:
    total = normal + warning + critical
    assert abs(total - 1.0) < 1e-9, "test probabilities must sum to 1"
    return {"NORMAL": normal, "WARNING": warning, "CRITICAL": critical}


def engine() -> AlertEngine:
    return AlertEngine(escalation=ESC)


# G-vocabulary mapping


def test_probability_of_aggregates_follow_the_g_mapping() -> None:
    p = probs(0.2, 0.5, 0.3)
    assert probability_of("WATCH_or_higher", p) == pytest.approx(0.8)  # 1 − P(NORMAL)
    assert probability_of("WARNING_or_higher", p) == pytest.approx(0.8)  # P(WARNING) + P(CRITICAL)
    assert probability_of("CRITICAL", p) == pytest.approx(0.3)


def test_probability_of_rejects_incomplete_or_unknown_targets() -> None:
    with pytest.raises(AlertEngineError):
        probability_of("WATCH_or_higher", {"NORMAL": 1.0})
    with pytest.raises(AlertEngineError):
        probability_of("NOT_A_LEVEL", probs(1.0, 0.0, 0.0))


# persistence: transitions fire only after consecutive windows


def test_green_to_watch_needs_three_consecutive_qualifying_windows() -> None:
    eng = engine()
    strong = probs(0.0, 0.9, 0.1)  # P(WATCH or higher) = 1.0 > 0.5
    assert eng.update("n1", strong) == "GREEN"
    assert eng.update("n1", strong) == "GREEN"
    assert eng.update("n1", strong) == "WATCH"  # fires on the 3rd consecutive


def test_streak_resets_when_a_window_stops_qualifying() -> None:
    eng = engine()
    strong = probs(0.0, 0.9, 0.1)
    weak = probs(0.99, 0.01, 0.0)
    assert eng.update("n1", strong) == "GREEN"
    assert eng.update("n1", strong) == "GREEN"
    assert eng.update("n1", weak) == "GREEN"  # streak broken
    assert eng.update("n1", strong) == "GREEN"
    assert eng.update("n1", strong) == "GREEN"
    assert eng.update("n1", strong) == "WATCH"  # only 3rd consecutive since reset


def test_watch_to_warning_fails_closed_without_required_conditions() -> None:
    eng = engine()
    eng.update("n1", probs(0.0, 0.95, 0.05))
    eng.update("n1", probs(0.0, 0.95, 0.05))
    eng.update("n1", probs(0.0, 0.95, 0.05))  # → WATCH
    strong = probs(0.0, 0.65, 0.35)  # P(WARNING or higher) = 1.0 > 0.6
    # no conditions at all, then only one of the two required:
    for cond in ({}, {"spatial_coherence_above_threshold": True}, {"displacement_trend_positive": True}):
        assert eng.update("n1", strong, conditions=cond) == "WATCH"
    # both conditions, three consecutive qualifying windows:
    both = {"spatial_coherence_above_threshold": True, "displacement_trend_positive": True}
    assert eng.update("n1", strong, conditions=both) == "WATCH"
    assert eng.update("n1", strong, conditions=both) == "WATCH"
    assert eng.update("n1", strong, conditions=both) == "WARNING"


def test_warning_to_critical_needs_physics_and_two_neighbours() -> None:
    eng = engine()
    for _ in range(3):
        eng.update("n1", probs(0.0, 0.9, 0.1))
    for _ in range(3):
        eng.update("n1", probs(0.0, 0.4, 0.6), conditions={"spatial_coherence_above_threshold": True, "displacement_trend_positive": True})
    assert eng.state("n1").level == "WARNING"
    severe = probs(0.0, 0.2, 0.8)  # P(CRITICAL) = 0.8 > 0.7
    # physics low but only one neighbour confirming (< min_confirming_neighbours=2):
    assert eng.update("n1", severe, conditions={"physics_residual_low": True, "neighbour_confirmations": 1}) == "WARNING"
    # two consecutive fully-confirmed windows (persistence 2):
    assert eng.update("n1", severe, conditions=CONFIRMED) == "WARNING"
    assert eng.update("n1", severe, conditions=CONFIRMED) == "CRITICAL"


#: no single reading escalates straight to CRITICAL


def test_no_single_reading_can_escalate_to_critical() -> None:
    eng = engine()
    worst = probs(0.0, 0.0, 1.0)  # maximally severe reading
    first = eng.update("n1", worst, conditions=FULLY_CONFIRMED)
    assert first == "GREEN", ": persistence 3 — one window cannot even reach WATCH"
    assert eng.state("n1").level != "CRITICAL"


def test_minimum_path_to_critical_requires_every_transition_in_order() -> None:
    """3 windows → WATCH, 3 more → WARNING, 2 more → CRITICAL; never a skip."""
    eng = engine()
    worst = probs(0.0, 0.0, 1.0)
    levels = [eng.update("n1", worst, conditions=FULLY_CONFIRMED) for _ in range(8)]
    assert levels == ["GREEN", "GREEN", "WATCH", "WATCH", "WATCH", "WARNING", "WARNING", "CRITICAL"]


def test_firing_a_transition_earns_no_credit_for_the_next_one() -> None:
    eng = engine()
    worst = probs(0.0, 0.0, 1.0)
    for _ in range(3):
        eng.update("n1", worst, conditions=FULLY_CONFIRMED)
    st = eng.state("n1")
    assert st.level == "WATCH"
    assert st.streaks.get("WATCH_to_WARNING", 0) == 0, "the firing window earned no next-transition credit"
    levels = [eng.update("n1", worst, conditions=FULLY_CONFIRMED) for _ in range(3)]
    assert levels == ["WATCH", "WATCH", "WARNING"]  # needs 3 FRESH windows


def test_top_level_is_absorbing_until_t051_hysteresis() -> None:
    eng = engine()
    worst = probs(0.0, 0.0, 1.0)
    for _ in range(20):
        eng.update("n1", worst, conditions=FULLY_CONFIRMED)
    assert eng.state("n1").level == "CRITICAL"
    calm = probs(0.99, 0.01, 0.0)
    assert eng.update("n1", calm) == "CRITICAL"  # de-escalation is 


# input validation


def test_probabilities_must_be_complete_in_range_and_sum_to_one() -> None:
    eng = engine()
    with pytest.raises(AlertEngineError):
        eng.update("n1", {"NORMAL": 0.6, "WARNING": 0.4})  # missing CRITICAL
    with pytest.raises(AlertEngineError):
        eng.update("n1", probs(1.2, -0.2, 0.0))  # outside [0, 1]
    with pytest.raises(AlertEngineError):
        eng.update("n1", {"NORMAL": 0.5, "WARNING": 0.5, "CRITICAL": 0.5})  # sums to 1.5


def test_state_is_tracked_per_node() -> None:
    eng = engine()
    strong = probs(0.0, 0.9, 0.1)
    for _ in range(3):
        eng.update("n1", strong)
    assert eng.state("n1").level == "WATCH"
    assert eng.state("n2").level == "GREEN", "nodes are independent"
    assert eng.state("n3").level == "GREEN"  # untouched node defaults to GREEN


# configuration discipline 


def test_engine_reads_the_real_alerts_config() -> None:
    assert alerts_config()["risk_levels"] == ["GREEN", "WATCH", "WARNING", "CRITICAL"]
    esc = escalation_thresholds()
    assert set(esc) == {"GREEN_to_WATCH", "WATCH_to_WARNING", "WARNING_to_CRITICAL"}
    eng = AlertEngine()  # defaults straight from configs/alerts.yaml
    assert eng.levels == ["GREEN", "WATCH", "WARNING", "CRITICAL"]


def test_unknown_requirement_or_missing_transition_raises_at_init() -> None:
    bad_req = dict(ESC)
    bad_req["GREEN_to_WATCH"] = dict(ESC["GREEN_to_WATCH"], requires=["vibes"])
    with pytest.raises(AlertEngineError, match="vibes"):
        AlertEngine(escalation=bad_req)
    missing = {k: v for k, v in ESC.items() if k != "WATCH_to_WARNING"}
    with pytest.raises(AlertEngineError, match="WATCH_to_WARNING"):
        AlertEngine(escalation=missing)
    zero_persist = dict(ESC)
    zero_persist["GREEN_to_WATCH"] = dict(ESC["GREEN_to_WATCH"], persistence_windows=0)
    with pytest.raises(AlertEngineError, match="persistence_windows"):
        AlertEngine(escalation=zero_persist)
