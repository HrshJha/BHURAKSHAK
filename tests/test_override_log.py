""" tests — manual-override logging (operator ID, reason, timestamp; model output never suppressed)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.config import alerts_config
from src.risk.override_log import OverrideLog, OverrideLogError


def log(tmp_path, config: dict | None = None) -> OverrideLog:
    return OverrideLog(tmp_path / "overrides.jsonl", config=config)


def sample(**kw) -> dict:
    base = dict(
        operator_id="op-42",
        reason="known maintenance vibration",
        node_id="n3",
        model_level="WARNING",
        overridden_level="GREEN",
        node_state_at_override="WATCH",
    )
    base.update(kw)
    return base


def test_override_records_operator_reason_and_timestamp(tmp_path) -> None:
    lg = log(tmp_path)
    ts = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
    rec = lg.record_override(**sample(now=ts))
    assert rec.operator_id == "op-42"
    assert rec.reason == "known maintenance vibration"
    assert rec.timestamp == "2026-09-27T12:00:00+00:00"
    assert rec.override_id == 1


def test_model_output_is_always_recorded_never_suppressed(tmp_path) -> None:
    lg = log(tmp_path)
    rec = lg.record_override(**sample(model_level="WARNING", overridden_level="GREEN"))
    assert rec.model_level == "WARNING", "the underlying model output must persist ()"
    # and the log refuses to exist under a suppressing configuration:
    with pytest.raises(OverrideLogError, match="suppress"):
        OverrideLog(tmp_path / "other.jsonl", config={"suppresses_model_output": True})
    assert alerts_config()["manual_override"]["suppresses_model_output"] is False


def test_missing_operator_id_or_reason_raises(tmp_path) -> None:
    lg = log(tmp_path)
    with pytest.raises(OverrideLogError, match="operator ID"):
        lg.record_override(**sample(operator_id="  "))
    with pytest.raises(OverrideLogError, match="reason"):
        lg.record_override(**sample(reason=""))


def test_requirements_can_be_relaxed_only_by_config(tmp_path) -> None:
    lg = OverrideLog(
        tmp_path / "relaxed.jsonl",
        config={"requires_operator_id": False, "requires_reason": False},
    )
    rec = lg.record_override(operator_id="", reason="", node_id="n1", model_level="CRITICAL", overridden_level="WARNING")
    assert rec.operator_id == ""  # allowed only because config relaxed it


def test_log_is_append_only_and_survives_reload(tmp_path) -> None:
    path = tmp_path / "overrides.jsonl"
    lg1 = OverrideLog(path)
    lg1.record_override(**sample())
    lg1.record_override(**sample(node_id="n4", model_level="CRITICAL", overridden_level="WARNING"))
    lg2 = OverrideLog(path)  # fresh instance over the same file
    assert len(lg2.records) == 2
    assert [r.override_id for r in lg2.records] == [1, 2]
    assert lg2.records[1].node_id == "n4"
    # the file itself is line-delimited JSON, one record per line (append-only)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_history_for_filters_by_node(tmp_path) -> None:
    lg = log(tmp_path)
    lg.record_override(**sample(node_id="n3"))
    lg.record_override(**sample(node_id="n4", model_level="CRITICAL"))
    lg.record_override(**sample(node_id="n3"))
    assert [r.override_id for r in lg.history_for("n3")] == [1, 3]
    assert lg.history_for("n9") == []


def test_monotonic_ids_across_appends(tmp_path) -> None:
    lg = log(tmp_path)
    ids = [lg.record_override(**sample(node_id=f"n{i}")).override_id for i in range(5)]
    assert ids == [1, 2, 3, 4, 5]
