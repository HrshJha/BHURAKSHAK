""" tests — de-escalation hysteresis ( acceptance bullet 8)."""

from __future__ import annotations

import pytest

from src.risk.alert_engine import AlertEngineError
from src.risk.hysteresis import HysteresisEngine

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

FULLY_CONFIRMED = {
    "spatial_coherence_above_threshold": True,
    "displacement_trend_positive": True,
    "physics_residual_low": True,
    "neighbour_confirmations": 3,
}
BOTH_SPATIAL = {"spatial_coherence_above_threshold": True, "displacement_trend_positive": True}


def probs(normal: float, warning: float, critical: float) -> dict[str, float]:
    return {"NORMAL": normal, "WARNING": warning, "critical_or_typo": None} if False else {
        "NORMAL": normal,
        "WARNING": warning,
        "CRITICAL": critical,
    }


def calm(conditions: dict | None = None) -> tuple[dict[str, float], dict]:
    """A window that fails every escalation gate (pure NORMAL, no evidence)."""
    return probs(1.0, 0.0, 0.0), conditions or {}


def engine(multiplier: float | None = 1.5) -> HysteresisEngine:
    return HysteresisEngine(escalation=ESC, multiplier=multiplier)


def to_warning(eng: HysteresisEngine, node: str = "n1") -> None:
    for _ in range(3):
        eng.update(node, probs(0.0, 0.95, 0.05))  # → WATCH (persistence 3)
    for _ in range(3):
        eng.update(node, probs(0.0, 0.65, 0.35), conditions=BOTH_SPATIAL)  # → WARNING


# --- the multiplier discipline -------------------------------------------------


def test_de_escalation_needs_one_and_a_half_times_the_escalation_windows() -> None:
    eng = engine()
    assert eng.de_escalation_windows(1) == 5, "ceil(3 × 1.5) out of WATCH"
    assert eng.de_escalation_windows(2) == 5, "ceil(3 × 1.5) out of WARNING"
    assert eng.de_escalation_windows(3) == 3, "ceil(2 × 1.5) out of CRITICAL"
    assert all(
        eng.de_escalation_windows(i) > int(ESC[f"{eng.levels[i - 1]}_to_{eng.levels[i]}"]["persistence_windows"])
        for i in (1, 2, 3)
    ), "recovery must be strictly slower than alarm at each level"


def test_multiplier_below_one_is_rejected() -> None:
    with pytest.raises(AlertEngineError, match="multiplier"):
        HysteresisEngine(escalation=ESC, multiplier=0.9)


def test_config_default_multiplier_is_one_point_five() -> None:
    assert HysteresisEngine().multiplier == 1.5  # straight from configs/alerts.yaml


# --- one-way-at-a-time behaviour ------------------------------------------------


def test_de_escalation_drops_one_level_and_takes_the_full_streak() -> None:
    eng = engine()
    to_warning(eng)
    p, c = calm()
    assert [eng.update("n1", p, conditions=c) for _ in range(4)] == ["WARNING"] * 4  # streak 4/5
    assert eng.update("n1", p, conditions=c) == "WATCH"  # 5th consecutive (ceil(4.5)=5) → drop one level
    assert eng.state("n1").level == "WATCH"


def test_single_qualifying_window_resets_the_recovery_streak() -> None:
    eng = engine()
    to_warning(eng)
    p, c = calm()
    for _ in range(4):
        eng.update("n1", p, conditions=c)  # de-escalation streak 4/5
    assert eng.update("n1", probs(0.0, 0.95, 0.05), conditions=BOTH_SPATIAL) == "WARNING"  # qualifies → reset
    assert [eng.update("n1", p, conditions=c) for _ in range(4)] == ["WARNING"] * 4
    assert eng.update("n1", p, conditions=c) == "WATCH"  # needs 5 FRESH consecutive windows


def test_out_of_level_window_also_resets_escalation() -> None:
    eng = engine()
    to_warning(eng)
    p, c = calm()
    for _ in range(3):
        eng.update("n1", p, conditions=c)  # recovery 3/5, escalation streak reset each time
    assert eng.update("n1", probs(0.0, 0.4, 0.35), conditions=BOTH_SPATIAL) == "WARNING"
    # P(WARNING or higher) = 0.75 > 0.6 — escalation streak restarts at 1, no climb


def test_escalation_and_deescalation_are_mutually_exclusive_per_window() -> None:
    eng = engine()
    to_warning(eng)
    p, c = calm()
    for _ in range(2):
        eng.update("n1", p, conditions=c)
    assert eng.update("n1", probs(0.0, 0.95, 0.05), conditions=BOTH_SPATIAL) == "WARNING"  # escalation credit only
    st = eng.state("n1")
    assert st.streaks.get("deescalation", 0) == 0, "a qualifying window must not also count toward recovery"
    assert st.streaks.get("escalation", 0) == 0, "it holds WARNING but feeds the CRITICAL gate, not the WATCH→WARNING gate — no escalation credit"


# ---: a borderline oscillating input does not flap ---------------------------


def test_borderline_oscillation_holds_level_without_flapping() -> None:
    eng = engine()
    to_warning(eng)
    assert eng.state("n1").level == "WARNING"
    qualifying = probs(0.0, 0.65, 0.35)  # just above the WATCH→WARNING gate
    seq = [probs(1.0, 0.0, 0.0), qualifying] * 12  # alternate calm/severe
    levels = [eng.update("n1", p, conditions=BOTH_SPATIAL if p is qualifying else {}) for p in seq]
    assert set(levels) == {"WARNING"}, "alternating input must hold the level, not flap"
    assert eng.state("n1").level == "WARNING"


def test_oscillation_between_watch_and_warning_never_flaps() -> None:
    eng = engine()
    for _ in range(3):
        eng.update("n1", probs(0.0, 0.95, 0.05))  # WATCH
    qualifying = probs(0.0, 0.65, 0.35)
    seq = [probs(1.0, 0.0, 0.0), qualifying] * 12
    levels = [eng.update("n1", p, conditions=BOTH_SPATIAL if p is qualifying else {}) for p in seq]
    assert set(levels) == {"WATCH"}, "level must hold under oscillation"
    assert eng.state("n1").level == "WATCH"


def test_genuine_recovery_still_happens_when_conditions_stay_calm() -> None:
    eng = engine()
    to_warning(eng)
    p, c = calm()
    for _ in range(5):
        eng.update("n1", p, conditions=c)
    assert eng.state("n1").level == "WATCH"
    for _ in range(5):
        eng.update("n1", p, conditions=c)
    assert eng.state("n1").level == "GREEN"


def test_crITICAL_de_escalates_with_its_own_window_count() -> None:
    eng = engine()
    for _ in range(3):
        eng.update("n1", probs(0.0, 0.0, 1.0), conditions=FULLY_CONFIRMED)
    for _ in range(3):
        eng.update("n1", probs(0.0, 0.0, 1.0), conditions=FULLY_CONFIRMED)
    for _ in range(2):
        eng.update("n1", probs(0.0, 0.0, 1.0), conditions=FULLY_CONFIRMED)
    assert eng.state("n1").level == "CRITICAL"
    p, c = calm()
    assert [eng.update("n1", p, conditions=c) for _ in range(2)] == ["CRITICAL", "CRITICAL"]
    assert eng.update("n1", p, conditions=c) == "WARNING"  # ceil(2×1.5)=3


def test_escalation_semantics_match_t050_alert_engine() -> None:
    """Same transition table → identical escalation history (single implementation of gates)."""
    eng = HysteresisEngine(escalation=ESC, multiplier=1.5)
    plain = __import__("src.risk.alert_engine", fromlist=["AlertEngine"]).AlertEngine(escalation=ESC)
    worst = probs(0.0, 0.0, 1.0)
    h_levels = [eng.update("n1", worst, conditions=FULLY_CONFIRMED) for _ in range(8)]
    a_levels = [plain.update("n1", worst, conditions=FULLY_CONFIRMED) for _ in range(8)]
    assert h_levels == a_levels == ["GREEN", "GREEN", "WATCH", "WATCH", "WATCH", "WARNING", "WARNING", "CRITICAL"]


def test_default_construction_reads_the_real_config() -> None:
    eng = HysteresisEngine()
    assert eng.levels == ["GREEN", "WATCH", "WARNING", "CRITICAL"]
    with pytest.raises(AlertEngineError, match="missing escalation transition"):
        HysteresisEngine(escalation={"GREEN_to_WATCH": ESC["GREEN_to_WATCH"]})
