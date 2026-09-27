"""Tabletop ground-truth protocol harness — PRD §23.1 (T-072).

The §23.1 tabletop rig validates the pipeline chain **against a physical
ground-truth protocol**: each trial's ``run_id`` (``trial_id`` in the
recorded data) links, end to end:

    rig actuator setting  →  independent reference measurement  →
    raw sensor data       →  derived risk state

Software harness only: operating the physical rig is out of workstream
scope (TASKS.md). The harness consumes RECORDED data in the schema produced
by ``scripts/generate_tabletop_dataset.py`` (the seeded stand-in campaign
lives in ``data/recorded/tabletop/`` — see its README); the SAME code runs
unchanged on a real campaign's CSVs.

Chain semantics (per trial):
- **Actuator setting**: the knob-turn schedule (M8 pitch 1.25 mm/turn) —
  ``max_displacement_mm_target`` metadata plus the per-window actuator
  displacement implied by the knob columns of the raw log.
- **Independent reference**: ``known_displacement_mm`` — mechanism truth at
  window end, measured OUTSIDE the sensor pipeline (the knob schedule), not
  derived from any sensor.
- **Raw sensor data**: the 10 Hz accelerometer / gyro / ToF / ultrasonic log.
- **Derived risk state**: the pipeline's windowed risk state computed from
  sensor features only.

The §23.1 acceptance quantity — **mesh-estimated vs reference displacement
error** — is computed here as window-level error statistics of the sensor
derived displacement vs ``known_displacement_mm``, with the honest caveat
that the rig's "mesh" is 4 nodes, 2 of which are pure references.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "TabletopProtocolError",
    "TrialLinkage",
    "link_trials",
    "derived_risk_state",
    "displacement_error_vs_reference",
    "evaluate_tabletop",
]

_KNOB_PITCH_MM = 1.25  # M8 thread pitch per full turn (plan §3.1)

_REQUIRED_METADATA = ("trial_id", "condition", "max_displacement_mm_target",
                      "achieved_displacement_mm_TD1", "achieved_displacement_mm_TD2")
_REQUIRED_WINDOWS = ("trial_id", "node_id", "window_start_ms", "displacement_mm",
                     "known_displacement_mm", "severity_class")


class TabletopProtocolError(ValueError):
    """Raised when recorded data breaks the §23.1 linkage contract."""


@dataclass
class TrialLinkage:
    """The §23.1 chain for one trial: actuator → reference → sensor → risk."""

    run_id: str
    condition: str
    #: Actuator setting: the configured target and what the knob achieved.
    actuator_target_mm: float
    actuator_achieved_mm: float
    #: Independent reference: mechanism truth OUTSIDE the sensor pipeline.
    reference_mm_max: float
    #: Raw sensor coverage for this trial.
    n_raw_samples: int
    n_raw_nodes: int
    n_windows: int
    #: Derived risk state summary (filled by :func:`derived_risk_state`).
    risk_state: dict = field(default_factory=dict)


def link_trials(
    metadata: pd.DataFrame,
    windowed: pd.DataFrame,
    raw: pd.DataFrame | None = None,
) -> list[TrialLinkage]:
    """Assert and return the §23.1 chain for every ``run_id``.

    Raises :class:`TabletopProtocolError` the moment the chain is broken:
    a trial in metadata without windows, windows without metadata, raw
    samples whose (trial, node) has no metadata, or duplicate (trial, node,
    window) keys.
    """
    for col in _REQUIRED_METADATA:
        if col not in metadata.columns:
            raise TabletopProtocolError(f"trial metadata missing {col!r}")
    for col in _REQUIRED_WINDOWS:
        if col not in windowed.columns:
            raise TabletopProtocolError(f"windowed data missing {col!r}")
    if metadata["trial_id"].duplicated().any():
        raise TabletopProtocolError("duplicate trial_id in metadata")
    if windowed.duplicated(subset=["trial_id", "node_id", "window_start_ms"]).any():
        raise TabletopProtocolError("duplicate (trial_id, node_id, window_start_ms) in windowed data")

    meta_ids = set(metadata["trial_id"])
    win_ids = set(windowed["trial_id"])
    if win_ids - meta_ids:
        raise TabletopProtocolError(f"windowed trials missing from metadata: {sorted(win_ids - meta_ids)[:5]}")
    if meta_ids - win_ids:
        raise TabletopProtocolError(f"metadata trials with no sensor windows: {sorted(meta_ids - win_ids)[:5]}")

    raw_ids: set[str] = set()
    if raw is not None:
        for col in ("trial_id", "node_id", "timestamp_ms"):
            if col not in raw.columns:
                raise TabletopProtocolError(f"raw sensor log missing {col!r}")
        raw_ids = set(raw["trial_id"].unique())
        orphans = raw_ids - meta_ids
        if orphans:
            raise TabletopProtocolError(f"raw-log trials missing from metadata: {sorted(orphans)[:5]}")
        if meta_ids - raw_ids:
            raise TabletopProtocolError(f"metadata trials absent from the raw log: {sorted(meta_ids - raw_ids)[:5]}")

    links = []
    for row in metadata.itertuples(index=False):
        win = windowed[windowed["trial_id"] == row.trial_id]
        ref_max = float(win["known_displacement_mm"].max())
        achieved = max(float(row.achieved_displacement_mm_TD1), float(row.achieved_displacement_mm_TD2))
        # §23.1 coherence: the reference can never exceed what the actuator
        # physically produced (knob schedule is the truth source).
        if ref_max > achieved + 1e-6:
            raise TabletopProtocolError(
                f"{row.trial_id}: reference displacement {ref_max:.2f} mm exceeds the "
                f"actuator-achieved {achieved:.2f} mm — linkage broken"
        )
        links.append(
            TrialLinkage(
                run_id=str(row.trial_id),
                condition=str(row.condition),
                actuator_target_mm=float(row.max_displacement_mm_target),
                actuator_achieved_mm=achieved,
                reference_mm_max=ref_max,
                n_raw_samples=int((raw["trial_id"] == row.trial_id).sum()) if raw is not None else 0,
                n_raw_nodes=int(raw.loc[raw["trial_id"] == row.trial_id, "node_id"].nunique()) if raw is not None else 0,
                n_windows=int(len(win)),
            )
        )
    return links


def derived_risk_state(
    windowed: pd.DataFrame,
    *,
    warning_mm: float = 15.0,
    critical_mm: float = 35.0,
) -> pd.DataFrame:
    """Per-window derived risk state from SENSOR features only.

    Thresholds are the plan §3.2 severity boundaries (0/15/35 mm) applied to
    the sensor-derived ``displacement_mm`` (ultrasonic minus baseline) — NOT
    to the reference. This is the honest analogue of the deployment rule
    "raise the alert when the sensor pipeline says so"; the reference is
    used only to SCORE this state, never to set it.
    """
    for col in ("displacement_mm", "known_displacement_mm", "severity_class"):
        if col not in windowed.columns:
            raise TabletopProtocolError(f"windowed data missing {col!r}")
    disp = windowed["displacement_mm"].to_numpy(dtype=float)
    state = np.where(disp > critical_mm, "CRITICAL",
                     np.where(disp > warning_mm, "WARNING", "NORMAL"))
    out = windowed.copy()
    out["derived_risk_state"] = state
    return out


def displacement_error_vs_reference(
    windowed: pd.DataFrame,
    *,
    nodes: tuple[str, ...] = ("N2", "N3"),
) -> dict[str, float]:
    """Mesh-estimated vs reference displacement error (§23.1 acceptance).

    ``nodes``: the active sensor nodes whose ``known_displacement_mm`` is
    the independent reference (N2 ← TD1, N3 ← TD2; N1/N4 are pure
    references with zero truth and are excluded by default).
    """
    for col in ("displacement_mm", "known_displacement_mm", "node_id"):
        if col not in windowed.columns:
            raise TabletopProtocolError(f"windowed data missing {col!r}")
    sel = windowed[windowed["node_id"].isin(nodes)]
    if sel.empty:
        raise TabletopProtocolError(f"no windows for reference nodes {nodes}")
    est = sel["displacement_mm"].to_numpy(dtype=float)
    ref = sel["known_displacement_mm"].to_numpy(dtype=float)
    err = est - ref
    return {
        "n_windows": int(len(sel)),
        "mae_mm": float(np.abs(err).mean()),
        "rmse_mm": float(np.sqrt(np.mean(err**2))),
        "max_abs_mm": float(np.abs(err).max()),
        "bias_mm": float(err.mean()),
    }


def evaluate_tabletop(
    metadata: pd.DataFrame,
    windowed: pd.DataFrame,
    raw: pd.DataFrame | None = None,
    *,
    warning_mm: float = 15.0,
    critical_mm: float = 35.0,
    active_nodes: tuple[str, ...] = ("N2", "N3"),
) -> dict:
    """Full §23.1 protocol run: link every trial, derive risk, score it.

    Returns a dict with the per-trial linkage records, the derived-risk
    confusion against the reference severity classes, and the §23.1
    displacement-error statistics.
    """
    links = link_trials(metadata, windowed, raw)
    state = derived_risk_state(windowed, warning_mm=warning_mm, critical_mm=critical_mm)
    sel = state[state["node_id"].isin(active_nodes)]

    ref_class = sel["severity_class"].to_numpy(dtype=int)
    est_class = sel["derived_risk_state"].map(
        {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
    ).to_numpy(dtype=int)
    # The rig's severity buckets are 0/1/2/3 (§3.2); collapse bucket 1 vs 2
    # (0–15 vs 15–35 mm) into one WARNING band for the 3-state comparison.
    ref_state = np.where(ref_class >= 3, 2, np.where(ref_class >= 1, np.where(
        sel["known_displacement_mm"].to_numpy(dtype=float) > critical_mm, 2, 1), 0))

    labels = ("NORMAL", "WARNING", "CRITICAL")
    cm = confusion_counts(ref_state, est_class, labels)
    per_trial = []
    for link in links:
        t = sel[sel["trial_id"] == link.run_id]
        per_trial.append(
            {
                "run_id": link.run_id,
                "condition": link.condition,
                "actuator_target_mm": link.actuator_target_mm,
                "actuator_achieved_mm": link.actuator_achieved_mm,
                "reference_mm_max": link.reference_mm_max,
                "n_windows": link.n_windows,
                "n_raw_samples": link.n_raw_samples,
                "n_raw_nodes": link.n_raw_nodes,
                "derived_state_max": (
                    str(t["derived_risk_state"].iloc[
                        np.argmax(t["displacement_mm"].to_numpy(dtype=float))
                    ]) if len(t) else "n/a"
                ),
            }
        )
    return {
        "n_trials": len(links),
        "trial_linkage": per_trial,
        "risk_state_confusion": cm,
        "displacement_error": displacement_error_vs_reference(state, nodes=active_nodes),
    }


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...]) -> dict:
    """Small 3-state confusion (dict-of-dicts) for the risk-state scoring."""
    out: dict[str, dict[str, int]] = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), strict=True):
        out[labels[t]][labels[p]] += 1
    return out
