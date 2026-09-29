"""De-escalation hysteresis — (de-escalation),.: de-escalation requires a *longer* persistence window than escalation
(default: one-and-a-half times the escalation window count at each level, from
configs/alerts.yaml ``de_escalation.persistence_multiplier``), so the system
does not flap between states on borderline readings — recovery is
deliberately slower than alarm.

Mechanism: a node holds its current level while the condition that promoted
it there keeps holding (the SAME aggregate ``P(target or higher) > threshold``
gate and the SAME ``requires`` evidence as the escalation transition out of
that level — no second set of thresholds to drift). A window that stops
qualifying starts a de-escalation streak; ``ceil(persistence × multiplier)``
CONSECUTIVE such windows drop the node one level. Any single qualifying
window resets the de-escalation streak (and a non-qualifying window resets
the escalation streak), so a borderline oscillating input can neither climb
nor recover — it holds steady, which is the intent.

Escalation and de-escalation are mutually exclusive per window: a window
either earns escalation credit or de-escalation credit, never both. Any level
change (either direction) resets all streaks. Like the escalation engine, level changes move AT MOST ONE level per update and a firing window
earns no credit for further movement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.config import deescalation_multiplier, escalation_thresholds
from src.risk.alert_engine import AlertEngineError, _conditions_met, probability_of

__all__ = ["HysteresisEngine", "NodeHysteresisState"]


@dataclass
class NodeHysteresisState:
    """Per-node state: current level, escalation and de-escalation streaks."""

    node_id: str
    level: str = "GREEN"
    streaks: dict[str, int] = field(default_factory=dict)


class HysteresisEngine:
    """ escalation + hysteresis de-escalation for one node population.

 Wraps the same transition table as 's AlertEngine (identical
 escalation behaviour) and adds the slower, persistence-multiplied
 de-escalation path. Levels and thresholds come from configs/alerts.yaml.
 """

    def __init__(
        self,
        escalation: Mapping[str, Mapping[str, Any]] | None = None,
        levels: list[str] | None = None,
        multiplier: float | None = None,
    ) -> None:
        self.escalation: dict[str, dict[str, Any]] = (
            {k: dict(v) for k, v in escalation.items()}
            if escalation is not None
            else {k: dict(v) for k, v in escalation_thresholds().items()}
        )
        self.levels = list(levels) if levels is not None else ["GREEN", "WATCH", "WARNING", "CRITICAL"]
        self.multiplier = float(multiplier if multiplier is not None else deescalation_multiplier())
        if self.multiplier < 1.0:
            raise AlertEngineError(
                f"de-escalation multiplier must be >= 1.0 (recovery slower than alarm), got {self.multiplier}"
            )
        if len(self.levels) < 2:
            raise AlertEngineError(f"need at least two alert levels, got {self.levels}")
        for l1, l2 in zip(self.levels, self.levels[1:]):
            if f"{l1}_to_{l2}" not in self.escalation:
                raise AlertEngineError(f"missing escalation transition {l1}_to_{l2!r}")
        self._nodes: dict[str, NodeHysteresisState] = {}

    def de_escalation_windows(self, level_index: int) -> int:
        """Persistence for dropping OUT of ``levels[level_index]`` (≥ its escalation count)."""
        import math

        key = f"{self.levels[level_index - 1]}_to_{self.levels[level_index]}"
        base = int(self.escalation[key]["persistence_windows"])
        return max(base, math.ceil(base * self.multiplier))

    def state(self, node_id: str) -> NodeHysteresisState:
        st = self._nodes.get(node_id)
        return st if st is not None else NodeHysteresisState(node_id=node_id, level=self.levels[0])

    def update(
        self,
        node_id: str,
        proba: Mapping[str, float],
        *,
        conditions: Mapping[str, Any] | None = None,
    ) -> str:
        """Feed one window; return the node's level after it (moves ≤ 1 level)."""
        conditions = conditions or {}
        st = self._nodes.setdefault(node_id, NodeHysteresisState(node_id=node_id, level=self.levels[0]))
        idx = self.levels.index(st.level)

        if idx < len(self.levels) - 1:  # escalation possible
            t = self.escalation[f"{self.levels[idx]}_to_{self.levels[idx + 1]}"]
            if probability_of(t["probability_of"], proba) > float(t["class_threshold"]) and _conditions_met(
                t, conditions
            ):
                st.streaks["escalation"] = st.streaks.get("escalation", 0) + 1
                st.streaks["deescalation"] = 0
                if st.streaks["escalation"] >= int(t["persistence_windows"]):
                    st.level = self.levels[idx + 1]
                    st.streaks.clear()
                return st.level
        st.streaks["escalation"] = 0

        if idx > 0:  # de-escalation possible (hysteresis: needs the longer streak)
            t = self.escalation[f"{self.levels[idx - 1]}_to_{self.levels[idx]}"]
            still_qualifies = probability_of(t["probability_of"], proba) > float(
                t["class_threshold"]
            ) and _conditions_met(t, conditions)
            if not still_qualifies:
                st.streaks["deescalation"] = st.streaks.get("deescalation", 0) + 1
                if st.streaks["deescalation"] >= self.de_escalation_windows(idx):
                    st.level = self.levels[idx - 1]
                    st.streaks.clear()
            else:
                st.streaks["deescalation"] = 0
        return st.level
