"""T-052 tests — §21.1 per-node → region roll-up (max rule + noisy-node guard)."""

from __future__ import annotations

import pytest

from src.config import alerts_config
from src.risk.aggregation import RegionAggregationError, RegionAggregator

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


def agg() -> RegionAggregator:
    return RegionAggregator(aggregation_cfg={"rule": "max", "single_node_critical_requires_neighbour_confirmation": True}, escalation=ESC)


# --- max rule -------------------------------------------------------------------


def test_region_state_is_the_max_over_node_states() -> None:
    a = agg()
    r = a.roll_up("panel-7", {"n1": "GREEN", "n2": "WATCH", "n3": "WARNING", "n4": "GREEN"})
    assert r.level == "WARNING"
    assert r.max_node == "n3"
    assert not r.guarded_single_node_critical


def test_a_severe_node_is_not_diluted_by_quiet_neighbours() -> None:
    a = agg()
    r = a.roll_up("panel-7", {"n1": "GREEN", "n2": "GREEN", "n3": "CRITICAL", "n4": "CRITICAL", "n5": "GREEN"})
    assert r.level == "CRITICAL", "two CRITICAL nodes confirm each other → region CRITICAL"
    assert r.critical_nodes == ["n3", "n4"]
    assert r.max_node == "n3"


def test_region_read_only_over_node_states() -> None:
    """Roll-up consumes a snapshot of node levels; it never mutates them."""
    levels = {"n1": "WARNING", "n2": "GREEN"}
    agg().roll_up("panel-7", levels)
    assert levels == {"n1": "WARNING", "n2": "GREEN"}


# --- single noisy node guard (§21.1 confirmation rule) ---------------------------


def test_single_unconfirmed_critical_node_cannot_make_the_region_critical() -> None:
    a = agg()
    r = a.roll_up("panel-7", {"n1": "GREEN", "n2": "GREEN", "n3": "CRITICAL"})
    assert r.level == "WARNING", "guarded one level below the max — elevated, not diluted"
    assert r.guarded_single_node_critical
    assert r.max_node == "n3"
    assert r.critical_nodes == ["n3"]


def test_single_critical_node_with_enough_neighbour_confirmation_is_honoured() -> None:
    a = agg()
    r = a.roll_up(
        "panel-7",
        {"n1": "GREEN", "n2": "GREEN", "n3": "CRITICAL"},
        neighbour_confirmations={"n3": 2},  # == min_confirming_neighbours
    )
    assert r.level == "CRITICAL"
    assert not r.guarded_single_node_critical


def test_one_confirming_neighbour_is_not_enough() -> None:
    a = agg()
    r = a.roll_up("panel-7", {"n1": "CRITICAL"}, neighbour_confirmations={"n1": 1})
    assert r.level == "WARNING", "config requires >= 2 confirming neighbours"
    assert r.guarded_single_node_critical


def test_guard_applies_only_to_the_top_level() -> None:
    a = agg()
    r = a.roll_up("panel-7", {"n1": "WARNING"})
    assert r.level == "WARNING" and not r.guarded_single_node_critical


def test_guard_can_be_disabled_by_config() -> None:
    a = RegionAggregator(
        aggregation_cfg={"rule": "max", "single_node_critical_requires_neighbour_confirmation": False},
        escalation=ESC,
    )
    assert a.roll_up("panel-7", {"n1": "CRITICAL"}).level == "CRITICAL"


# --- validation -------------------------------------------------------------------


def test_empty_region_and_unknown_levels_raise() -> None:
    a = agg()
    with pytest.raises(RegionAggregationError, match="at least one constituent node"):
        a.roll_up("panel-7", {})
    with pytest.raises(RegionAggregationError, match="unknown alert level"):
        a.roll_up("panel-7", {"n1": "ORANGE"})


def test_non_max_rule_is_rejected() -> None:
    with pytest.raises(RegionAggregationError, match="max"):
        RegionAggregator(aggregation_cfg={"rule": "mean"}, escalation=ESC)


def test_default_construction_reads_the_real_config() -> None:
    a = RegionAggregator()  # straight from configs/alerts.yaml
    assert a.rule == "max"
    assert a.single_node_critical_requires_confirmation is True
    assert a.levels == ["GREEN", "WATCH", "WARNING", "CRITICAL"]
    assert a.min_confirming_neighbours == alerts_config()["escalation"]["WARNING_to_CRITICAL"]["min_confirming_neighbours"] == 2
