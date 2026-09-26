"""Per-node → region risk roll-up — PRD §21.1 (spatial aggregation), T-052.

§21.1: risk state is tracked per-node and rolled up to a region/panel level
as the **max over its constituent nodes' states**, so one severely affected
node cannot be diluted by averaging with quiet neighbours — but a single
noisy node also cannot trigger a panel-wide CRITICAL without the
multi-node confirmation rule (the §21.1 WARNING→CRITICAL requirement:
confirmation by neighbouring nodes, ``min_confirming_neighbours`` from
configs/alerts.yaml).

Concretely, a region reads CRITICAL only when:
- at least TWO constituent nodes are at CRITICAL (the multi-node case IS the
  confirmation), or
- the CRITICAL node(s) report the configured number of confirming
  neighbours.

A lone, unconfirmed CRITICAL node degrades the REGION one level (WARNING):
the panel still shows elevated risk — never diluted to GREEN — while the
node itself keeps its true state (roll-up is read-only over node states).
The rule name and the guard flag are read from configs/alerts.yaml
(``spatial_aggregation``), never hard-coded (NFR-6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.config import alerts_config

__all__ = ["RegionAggregationError", "RegionAggregation", "RegionAggregator"]


class RegionAggregationError(ValueError):
    """Raised on invalid roll-up inputs or configuration."""


@dataclass
class RegionAggregation:
    """Result of one roll-up: region level plus the evidence behind it."""

    region_id: str
    level: str
    #: the node holding the max per-node level (None for an empty region)
    max_node: str | None
    #: nodes at CRITICAL that drove (or were guarded at) the region level
    critical_nodes: list[str]
    #: True when the single-node CRITICAL guard demoted the region a level
    guarded_single_node_critical: bool


class RegionAggregator:
    """Max-rule roll-up of per-node §21.1 alert levels to a region/panel state."""

    def __init__(
        self,
        aggregation_cfg: Mapping[str, Any] | None = None,
        escalation: Mapping[str, Mapping[str, Any]] | None = None,
        levels: list[str] | None = None,
    ) -> None:
        cfg = alerts_config()["spatial_aggregation"] if aggregation_cfg is None else dict(aggregation_cfg)
        self.rule = str(cfg.get("rule", "max"))
        if self.rule != "max":
            raise RegionAggregationError(f"unsupported spatial aggregation rule {self.rule!r} (§21.1 mandates max)")
        self.single_node_critical_requires_confirmation = bool(
            cfg.get("single_node_critical_requires_neighbour_confirmation", True)
        )
        self.levels = list(levels) if levels is not None else list(alerts_config()["risk_levels"])
        esc = escalation if escalation is not None else alerts_config()["escalation"]
        self.min_confirming_neighbours = int(esc["WARNING_to_CRITICAL"].get("min_confirming_neighbours", 1))

    def _rank(self, level: str) -> int:
        try:
            return self.levels.index(level)
        except ValueError:
            raise RegionAggregationError(f"unknown alert level {level!r} (expected one of {self.levels})") from None

    def roll_up(
        self,
        region_id: str,
        node_levels: Mapping[str, str],
        neighbour_confirmations: Mapping[str, int] | None = None,
    ) -> RegionAggregation:
        """Aggregate node levels for ``region_id``.

        ``node_levels`` maps node_id → §21.1 alert level (as produced by the
        T-050/T-051 engines). ``neighbour_confirmations`` optionally maps
        node_id → confirming-neighbour count for the single-node CRITICAL
        guard; nodes absent from it count as unconfirmed.
        """
        if not node_levels:
            raise RegionAggregationError("a region needs at least one constituent node to roll up")
        confirmations = neighbour_confirmations or {}
        for node, level in node_levels.items():
            self._rank(level)  # validate early, with a clear error

        critical_nodes = sorted(n for n, lv in node_levels.items() if lv == self.levels[-1])
        max_node = max(node_levels, key=lambda n: self._rank(node_levels[n]))
        max_rank = self._rank(node_levels[max_node])
        region_rank = max_rank
        guarded = False

        if (
            self.single_node_critical_requires_confirmation
            and max_rank == len(self.levels) - 1
            and len(critical_nodes) < 2
        ):
            confirmed = any(
                int(confirmations.get(node, 0)) >= self.min_confirming_neighbours for node in critical_nodes
            )
            if not confirmed:
                region_rank = max_rank - 1  # de-graded, NOT diluted: one level below the max
                guarded = True

        return RegionAggregation(
            region_id=region_id,
            level=self.levels[region_rank],
            max_node=max_node,
            critical_nodes=critical_nodes,
            guarded_single_node_critical=guarded,
        )
