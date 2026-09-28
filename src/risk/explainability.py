"""Explainability contribution breakdown — PRD §22, FR-11 (T-054).

§22/FR-11: the system must never present a bare probability ("AI = 92%")
without a contributing-signal breakdown — a functional requirement, not a UI
nicety, because a black-box percentage is not actionable or defensible to a
mine operator. This module makes FR-11 structural:

- :func:`emit_risk_output` is the sanctioned way to emit a risk output; it
  returns the level, the probabilities and the explanation TOGETHER, and
  refuses (raises) when no contributing signals are supplied;
- :class:`RiskExplanation` renders §22-style lines ("+ high tilt velocity",
  "+ 5 neighbouring nodes anomalous", "+ InSAR deformation agreement",
  "+ physics residual low") with the numeric evidence kept alongside for
  auditability.

Signal semantics come from a fixed catalogue of canonical §22 signals:
most raise risk when positive; ``physics_residual`` is the §21 special case
— a residual near zero (supplied normalised by the physics model's noise
scale, i.e. a z-score) CONFIRMS the classification ("physically consistent")
while a large residual suggests a sensor artifact and counts against the
output. Signals outside the catalogue are included with an honest,
direction-neutral bullet rather than a guessed influence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from src.risk.alert_engine import AlertEngineError, MODEL_CLASSES, probability_of

__all__ = ["ExplanationError", "SignalContribution", "RiskExplanation", "SIGNAL_CATALOGUE", "emit_risk_output"]

#: normalised |physics_residual| at or below this counts as §21 "physically consistent"
RESIDUAL_Z_OK = 1.0


class ExplanationError(ValueError):
    """Raised when a risk output cannot be honestly explained."""


@dataclass
class SignalContribution:
    """One contributing signal: its value, influence direction and §22 phrase."""

    signal: str
    value: float
    raises_risk: bool | None  # None = unknown/neutral signal, direction not claimed
    text: str


@dataclass
class RiskExplanation:
    """A §22/FR-11 breakdown attached to one risk output. Never bare."""

    node_id: str
    predicted_level: str
    probabilities: dict[str, float]
    contributions: list[SignalContribution] = field(default_factory=list)

    @property
    def summary(self) -> list[str]:
        """§22-style lines: '+' raises risk, '−' counts against, '·' neutral."""
        lines = []
        for c in self.contributions:
            sign = "+" if c.raises_risk else ("-" if c.raises_risk is False else "·")
            lines.append(f"{sign} {c.text} ({c.signal}={c.value:.4g})")
        return lines

    def render(self) -> str:
        """Full dashboard block: headline probability + the §22 breakdown."""
        head = f"Risk: {self.predicted_level} (P={self.probabilities[self.predicted_level]:.2f})"
        return "\n".join([head, *self.summary])


#: canonical §22 signal → (raises_when, phrase when raising, phrase when not)
SIGNAL_CATALOGUE: dict[str, tuple[str, str, str]] = {
    "tilt_velocity": ("positive", "high tilt velocity", "tilt velocity quiet"),
    "displacement_trend": ("positive", "increasing displacement", "no sustained displacement increase"),
    "neighbour_anomaly_count": ("positive", "neighbouring nodes anomalous", "no neighbouring nodes anomalous"),
    "spatial_coherence": ("positive", "high spatial coherence across the mesh", "low spatial coherence"),
    "insar_agreement": ("positive", "InSAR deformation agreement", "no InSAR agreement available"),
    "anomaly_score": ("positive", "isolation-forest anomaly score elevated", "anomaly score unremarkable"),
    "vibration_rms": ("positive", "vibration burst detected", "vibration unremarkable"),
    "physics_residual": (
        "near_zero",
        "physics residual low — physically consistent",
        "physics residual large — physics disagrees, possible sensor artifact",
    ),
}


def _contribution(signal: str, value: float) -> SignalContribution:
    entry = SIGNAL_CATALOGUE.get(signal)
    if entry is None:
        return SignalContribution(signal=signal, value=value, raises_risk=None, text=f"signal reported: {signal}")
    mode, raising, reassuring = entry
    if mode == "near_zero":
        ok = abs(value) <= RESIDUAL_Z_OK
        text = f"{raising} (z={value:+.2f})" if ok else f"{reassuring} (z={value:+.2f})"
        return SignalContribution(signal=signal, value=value, raises_risk=ok, text=text)
    raising_now = value > 0
    text = raising if raising_now else reassuring
    if signal == "neighbour_anomaly_count" and raising_now:
        text = f"{value:.0f} {raising}"
    return SignalContribution(signal=signal, value=value, raises_risk=raising_now, text=text)


def _validated_proba(proba: Mapping[str, float]) -> dict[str, float]:
    missing = [c for c in MODEL_CLASSES if c not in proba]
    if missing:
        raise ExplanationError(f"probabilities missing model classes {missing}")
    values = {c: float(proba[c]) for c in MODEL_CLASSES}
    for c, v in values.items():
        if not 0.0 <= v <= 1.0:
            raise ExplanationError(f"P({c})={v} outside [0, 1]")
    if abs(sum(values.values()) - 1.0) > 1e-4:
        raise ExplanationError("class probabilities must sum to 1 (§15)")
    return values


def emit_risk_output(
    node_id: str,
    proba: Mapping[str, float],
    signals: Mapping[str, float],
) -> tuple[str, dict[str, float], RiskExplanation]:
    """FR-11 emission gate: level + probabilities + breakdown, never a bare probability.

    Raises :class:`ExplanationError` when ``signals`` is empty — a risk
    output without a contributing-signal breakdown cannot leave this module.
    """
    values = _validated_proba(proba)
    if not signals:
        raise ExplanationError(
            "FR-11/§22: refusing to emit a bare probability without a contributing-signal "
            "breakdown — supply at least one contributing signal"
        )
    contributions = [_contribution(name, float(value)) for name, value in sorted(signals.items())]

    # deterministic argmax; ties resolve to the more severe class
    severity = {c: i for i, c in enumerate(MODEL_CLASSES)}  # NORMAL < WARNING < CRITICAL
    level = max(MODEL_CLASSES, key=lambda c: (values[c], severity[c]))

    explanation = RiskExplanation(
        node_id=node_id,
        predicted_level=level,
        probabilities=values,
        contributions=contributions,
    )
    return level, values, explanation


# keep the §21.1 aggregate-probability helper importable from here too, so
# dashboard code can show P(WATCH or higher) WITH its breakdown from one import
__all__.append("probability_of")
_ = AlertEngineError  # re-exported for callers that unify error handling
