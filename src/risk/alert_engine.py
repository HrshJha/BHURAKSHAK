"""Apply configured persistence and evidence rules to risk predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.config import alerts_config, escalation_thresholds

__all__ = ["AlertEngineError", "NodeAlertState", "AlertEngine", "probability_of", "CONDITION_KEYS"]

#: The MVP model classes whose probabilities the engine consumes.
MODEL_CLASSES = ("NORMAL", "WARNING", "CRITICAL")

#: requirement name (configs/alerts.yaml) → the condition key that evidences it.
CONDITION_KEYS: dict[str, str] = {
    "spatial_coherence_above_threshold": "spatial_coherence_above_threshold",
    "displacement_trend_positive": "displacement_trend_positive",
    "physics_residual_low": "physics_residual_low",
    "neighbour_confirmation": "neighbour_confirmations",
}


class AlertEngineError(ValueError):
    """Raised on invalid alert-engine inputs or configuration."""


def probability_of(target: str, proba: Mapping[str, float]) -> float:
    """ aggregate ``P(target or higher)`` under the G-vocabulary mapping.

 ``target`` is the ``probability_of`` value from configs/alerts.yaml
 (``WATCH_or_higher``, ``WARNING_or_higher`` or ``CRITICAL``).
 """
    missing = [c for c in MODEL_CLASSES if c not in proba]
    if missing:
        raise AlertEngineError(f"probabilities missing model classes {missing}")
    if target == "CRITICAL":
        return float(proba["CRITICAL"])
    if target == "WARNING_or_higher":
        return float(proba["WARNING"]) + float(proba["CRITICAL"])
    if target == "WATCH_or_higher":
        return 1.0 - float(proba["NORMAL"])
    raise AlertEngineError(f"unsupported probability_of target {target!r}")


def _conditions_met(transition_cfg: Mapping[str, Any], conditions: Mapping[str, Any]) -> bool:
    """Evaluate a transition's ``requires`` list; fail closed on missing evidence."""
    for req in transition_cfg.get("requires", []):
        key = CONDITION_KEYS.get(req)
        if key is None:
            raise AlertEngineError(f"unknown escalation requirement {req!r} in configs/alerts.yaml")
        if req == "neighbour_confirmation":
            need = int(transition_cfg.get("min_confirming_neighbours", 1))
            got = int(conditions.get(key, 0))
            if got < need:
                return False
        elif not bool(conditions.get(key, False)):
            return False
    return True


@dataclass
class NodeAlertState:
    """ state for one node: current level + consecutive-window streaks."""

    node_id: str
    level: str = "GREEN"
    #: streak of consecutive qualifying windows per transition key
    streaks: dict[str, int] = field(default_factory=dict)


class AlertEngine:
    """Per-node escalation state machine, fully configuration-driven."""

    def __init__(
        self,
        escalation: Mapping[str, Mapping[str, Any]] | None = None,
        levels: list[str] | None = None,
    ) -> None:
        self.escalation: dict[str, dict[str, Any]] = (
            {k: dict(v) for k, v in escalation.items()}
            if escalation is not None
            else {k: dict(v) for k, v in escalation_thresholds().items()}
        )
        self.levels = list(levels) if levels is not None else list(alerts_config()["risk_levels"])
        self._validate_config()
        self._nodes: dict[str, NodeAlertState] = {}

    def _validate_config(self) -> None:
        if len(self.levels) < 2:
            raise AlertEngineError(f"need at least two alert levels, got {self.levels}")
        for l1, l2 in zip(self.levels, self.levels[1:]):
            key = f"{l1}_to_{l2}"
            t = self.escalation.get(key)
            if t is None:
                raise AlertEngineError(f"missing escalation transition {key!r}")
            if int(t.get("persistence_windows", 0)) < 1:
                raise AlertEngineError(f"transition {key!r} needs persistence_windows >= 1")
            for req in t.get("requires", []):
                if req not in CONDITION_KEYS:
                    raise AlertEngineError(f"unknown escalation requirement {req!r}")

    def state(self, node_id: str) -> NodeAlertState:
        """Current state for ``node_id`` (GREEN before the node's first update)."""
        st = self._nodes.get(node_id)
        if st is None:
            return NodeAlertState(node_id=node_id, level=self.levels[0])
        return st

    def _validate_proba(self, proba: Mapping[str, float]) -> None:
        missing = [c for c in MODEL_CLASSES if c not in proba]
        if missing:
            raise AlertEngineError(f"probabilities missing model classes {missing}")
        values = {c: float(proba[c]) for c in MODEL_CLASSES}
        for c, v in values.items():
            if not 0.0 <= v <= 1.0:
                raise AlertEngineError(f"P({c})={v} outside [0, 1]")
        total = sum(values.values())
        if abs(total - 1.0) > 1e-4:
            raise AlertEngineError(f"class probabilities must sum to 1 (), got {total}")

    def update(
        self,
        node_id: str,
        proba: Mapping[str, float],
        *,
        conditions: Mapping[str, Any] | None = None,
    ) -> str:
        """Feed one window; return the node's alert level after it.

 ``conditions`` evidences the configured ``requires`` items (booleans,
 ``neighbour_confirmations`` as a count); missing evidence fails
 closed. At most ONE escalation fires per update, and the window that
 fires a transition earns no streak credit for the next one — the
 conservative reading of.
 """
        conditions = conditions or {}
        self._validate_proba(proba)
        st = self._nodes.setdefault(node_id, NodeAlertState(node_id=node_id, level=self.levels[0]))
        idx = self.levels.index(st.level)
        if idx == len(self.levels) - 1:
            return st.level  # top level is absorbing here; de-escalation is 

        l1, l2 = self.levels[idx], self.levels[idx + 1]
        key = f"{l1}_to_{l2}"
        t = self.escalation[key]
        qualifies = probability_of(t["probability_of"], proba) > float(t["class_threshold"]) and _conditions_met(
            t, conditions
        )
        st.streaks[key] = st.streaks.get(key, 0) + 1 if qualifies else 0
        if st.streaks[key] >= int(t["persistence_windows"]):
            st.level = l2
            for k in st.streaks:
                st.streaks[k] = 0
        return st.level
