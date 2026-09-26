"""Operator override logging — PRD §21.1 (manual override), T-053.

§21.1: an operator can force an acknowledgement/hold at a given level from
the dashboard (e.g. during known maintenance or vibration events), logged
with **operator ID and reason** — but they **cannot suppress the underlying
model output from being recorded**. Every override therefore persists:

- who (``operator_id``) and why (``reason``) — both required by config
  (``manual_override.requires_operator_id`` / ``requires_reason``);
- what the model said (the recorded model level) and what the operator held
  it at (``overridden_level``), plus the node/region it applies to;
- a monotonic UTC timestamp.

The log is append-only (JSON Lines): records are never edited or removed,
so the audit trail survives the session. Every entry carries the model's own
level — suppression is structurally impossible because the record cannot be
created without it (``suppresses_model_output`` is asserted False against
configs/alerts.yaml at construction).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import alerts_config

__all__ = ["OverrideLogError", "OverrideRecord", "OverrideLog"]


class OverrideLogError(ValueError):
    """Raised on invalid override records or configuration."""


@dataclass
class OverrideRecord:
    """One §21.1 manual-override entry (append-only, never edited)."""

    override_id: int
    timestamp: str  # ISO-8601 UTC
    operator_id: str
    reason: str
    node_id: str
    model_level: str  # the model's own output — ALWAYS recorded (§21.1)
    overridden_level: str  # the level the operator held it at
    node_state_at_override: str | None = None  # engine state, when supplied

    def to_dict(self) -> dict[str, Any]:
        return {
            "override_id": self.override_id,
            "timestamp": self.timestamp,
            "operator_id": self.operator_id,
            "reason": self.reason,
            "node_id": self.node_id,
            "model_level": self.model_level,
            "overridden_level": self.overridden_level,
            "node_state_at_override": self.node_state_at_override,
        }


class OverrideLog:
    """Append-only §21.1 override log with JSONL persistence."""

    def __init__(self, path: str | Path, config: dict[str, Any] | None = None) -> None:
        cfg = alerts_config()["manual_override"] if config is None else config
        if bool(cfg.get("suppresses_model_output", False)):
            raise OverrideLogError(
                "manual_override.suppresses_model_output must be false — §21.1 forbids "
                "suppressing the underlying model output"
            )
        self.requires_operator_id = bool(cfg.get("requires_operator_id", True))
        self.requires_reason = bool(cfg.get("requires_reason", True))
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: list[OverrideRecord] = self._load()

    def _load(self) -> list[OverrideRecord]:
        records: list[OverrideRecord] = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(OverrideRecord(**json.loads(line)))
        return records

    @property
    def records(self) -> list[OverrideRecord]:
        return list(self._records)

    def record_override(
        self,
        *,
        operator_id: str,
        reason: str,
        node_id: str,
        model_level: str,
        overridden_level: str,
        node_state_at_override: str | None = None,
        now: datetime | None = None,
    ) -> OverrideRecord:
        """Append one override; the model level is mandatory (never suppressed)."""
        if self.requires_operator_id and not str(operator_id).strip():
            raise OverrideLogError("operator ID is required for a manual override (configs/alerts.yaml)")
        if self.requires_reason and not str(reason).strip():
            raise OverrideLogError("a reason is required for a manual override (configs/alerts.yaml)")
        if not str(node_id).strip():
            raise OverrideLogError("override must name the node/region it applies to")
        if not str(model_level).strip():
            raise OverrideLogError("the model's own level must be recorded — §21.1 forbids suppressing it")
        if not str(overridden_level).strip():
            raise OverrideLogError("the overridden level must be recorded")
        ts = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        with self._lock:
            rec = OverrideRecord(
                override_id=len(self._records) + 1,
                timestamp=ts,
                operator_id=str(operator_id),
                reason=str(reason),
                node_id=str(node_id),
                model_level=str(model_level),
                overridden_level=str(overridden_level),
                node_state_at_override=node_state_at_override,
            )
            self._records.append(rec)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")
        return rec

    def history_for(self, node_id: str) -> list[OverrideRecord]:
        """All overrides recorded for one node/region, in log order."""
        return [r for r in self._records if r.node_id == node_id]
