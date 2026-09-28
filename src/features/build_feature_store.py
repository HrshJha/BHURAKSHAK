"""Feature store assembly — PRD §13 (T-042).

Assembles the first-iteration feature matrix from Groups A–F over the §10
windowing of a raw node table, and **asserts the §13 budget: 40–70 features
inclusive** — failing loudly (``FeatureBudgetError``) if the assembled count
falls outside the budget.

Feature-count accounting (deterministic): the §13-exact names from Groups A,
D, E, F plus Group C's eight §13 names, plus Group B's §13 names per channel
(10 features × 3 movement channels) + the §13-exact bare set. Groups G–J are
out of first-iteration scope (G/H arrive with Phase-4 modalities; J is
ablation-gated per §13/§38).

Also emits a model-input frame: window keys + selected features + the §12
labels (``anomaly_label`` majority, ``risk_label`` majority,
``progression_label`` majority) for training. The guard in
:func:`src.features.windowing.assert_windowed` is applied to the model input
so raw rows can never flow downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.group_a_physical import GROUP_A_FEATURES, emit_group_a
from src.features.group_b_temporal import GROUP_B_FEATURES, emit_group_b
from src.features.group_c_spatial import GROUP_C_FEATURES, emit_group_c
from src.features.group_d_vibration import GROUP_D_FEATURES, emit_group_d
from src.features.group_e_health import GROUP_E_FEATURES, emit_group_e
from src.features.group_f_physics import GROUP_F_FEATURES, emit_group_f
from src.features.provenance import assert_registered
from src.features.windowing import WINDOW_TIMESTAMP, assert_windowed, build_windows

__all__ = [
    "FeatureBudgetError",
    "FeatureStoreReport",
    "feature_names",
    "check_feature_budget",
    "build_feature_store",
]

#: Labels carried onto the model-input frame (§12 — kept separate).
MODEL_LABELS = ("anomaly_label", "risk_label", "progression_label", "fault_label")

#: Group B per-channel part of the budget: tilt_x/tilt_y get explicit
#: per-channel features; the §13-exact bare temporal set IS the displacement
#: channel's view (the bare names alias displacement_* in emit_group_b), so it
#: is counted once — no duplication, and the budget stays honest.
B_PER_CHANNEL = ("tilt_x", "tilt_y")


class FeatureBudgetError(RuntimeError):
    """Raised when the assembled feature count violates §13's 40–70 budget."""


@dataclass
class FeatureStoreReport:
    n_windows: int
    features: list[str]
    per_group: dict[str, list[str]]
    budget_ok: bool
    center_mode: str
    gated_features: list[str]


def feature_names(b_channels: tuple[str, ...] = B_PER_CHANNEL) -> list[str]:
    """The deterministic first-iteration feature list (Groups A–F).

    Accounting: A=5, B=10×2 per-channel + 10 bare (the displacement view) = 30,
    C=8, D=7, E=7, F=4 → 61 features, inside §13's 40–70 budget.
    """
    names = list(GROUP_A_FEATURES)
    names += [f"{ch}_{feat}" for ch in b_channels for feat in GROUP_B_FEATURES]
    names += list(GROUP_B_FEATURES)  # §13-exact bare set (displacement channel)
    names += list(GROUP_C_FEATURES)
    names += list(GROUP_D_FEATURES)
    names += list(GROUP_E_FEATURES)
    names += list(GROUP_F_FEATURES)
    # de-duplicate, preserving order
    return list(dict.fromkeys(names))


def check_feature_budget(names: list[str], *, minimum: int = 40, maximum: int = 70) -> None:
    """§13: raise unless the feature count is within [40, 70]."""
    n = len(set(names))
    if not minimum <= n <= maximum:
        raise FeatureBudgetError(
            f"§13 feature budget violated: {n} features assembled, allowed {minimum}–{maximum}. "
            "First iteration must stay within the budget — reduce or extend feature groups."
        )


def _majority(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return np.nan
    return values.mode().iloc[0]


def build_feature_store(
    raw: pd.DataFrame,
    node_coords: pd.DataFrame,
    *,
    window_channels: tuple[str, ...] | None = None,
    center_mode: str = "detected",
) -> tuple[pd.DataFrame, FeatureStoreReport]:
    """Build the model-input feature store from a raw node table.

    Returns ``(model_input, report)`` where ``model_input`` carries structural
    keys, features, and labels separately. Emitters receive a label-blind copy
    of the input. Oracle geometry is refused for feature builds; diagnostics
    must use the separate diagnostic tooling, never a model store.
    """
    if center_mode == "oracle":
        raise ValueError("center_mode='oracle' is forbidden for model feature stores")
    if center_mode != "detected":
        raise ValueError(f"unsupported center_mode {center_mode!r}")
    # Keep only structural grouping keys and observable channels. Labels,
    # scenario descriptors, simulator truth, and injected parameters never
    # reach a feature emitter.
    structural = [c for c in ("event_id", "node_id", "timestamp") if c in raw.columns]
    known_channels = (
        "tilt_x", "tilt_y", "tilt_magnitude", "displacement", "strain",
        "vibration_rms", "vibration_peak", "battery", "RSSI", "SNR", "packet_loss",
    )
    observable = [c for c in known_channels if c in raw.columns]
    feature_raw = raw[structural + observable].copy()
    windowed = build_windows(feature_raw, channels=window_channels, labels=()).df

    # ---- per-group emission -------------------------------------------------
    group_a = emit_group_a(windowed)
    group_b = emit_group_b(windowed)
    group_c = emit_group_c(windowed, node_coords, center_mode=center_mode)
    group_d = emit_group_d(windowed)
    group_e = emit_group_e(windowed, feature_raw)
    group_f = emit_group_f(windowed, node_coords)

    keys = ["event_id", "node_id", "window_index", WINDOW_TIMESTAMP]
    model = group_a[keys + list(GROUP_A_FEATURES)]

    def _prefix_merge(base: pd.DataFrame, add: pd.DataFrame, names: list[str]) -> pd.DataFrame:
        cols = keys + [c for c in add.columns if c in names]
        return base.merge(add[cols], on=keys, how="left")

    model = _prefix_merge(model, group_b, feature_names())
    model = _prefix_merge(model, group_c, list(GROUP_C_FEATURES) + ["center_mode", "spatial_gate_reason"])
    model = _prefix_merge(model, group_d, list(GROUP_D_FEATURES))
    model = _prefix_merge(model, group_e, list(GROUP_E_FEATURES))
    model = _prefix_merge(model, group_f, list(GROUP_F_FEATURES))
# 
    present_labels = [lab for lab in MODEL_LABELS if lab in raw.columns]
    if present_labels:
        label_rows = raw[["event_id", "node_id", "timestamp", *present_labels]].copy()
        label_rows["window_index"] = _window_index_of(label_rows)
        label_rows = label_rows[label_rows["window_index"] >= 0]
        aggregations = {
            lab: (_majority if lab != "fault_label" else "first") for lab in present_labels
        }
        label_frame = (
            label_rows.groupby(["event_id", "node_id", "window_index"], sort=False)
            .agg(aggregations)
            .reset_index()
        )
        model = model.merge(label_frame, on=["event_id", "node_id", "window_index"], how="left")

    names = feature_names()
    assert_registered(names)
    missing = [f for f in names if f not in model.columns]
    if missing:
        raise FeatureBudgetError(f"feature store assembly failed — features not emitted: {missing}")

    assert_windowed(model)  # guard: model input must be window-level, never raw
    check_feature_budget(names)
    from src.features.schema_guard import assert_schema_drift

    assert_schema_drift(model.columns)
    report = FeatureStoreReport(
        n_windows=len(model),
        features=names,
        per_group={
            "A_physical": list(GROUP_A_FEATURES),
            "B_temporal": [f"{ch}_{feat}" for ch in B_PER_CHANNEL for feat in GROUP_B_FEATURES]
            + list(GROUP_B_FEATURES),
            "C_spatial": list(GROUP_C_FEATURES),
            "D_vibration": list(GROUP_D_FEATURES),
            "E_sensor_health": list(GROUP_E_FEATURES),
            "F_physics": list(GROUP_F_FEATURES),
        },
        budget_ok=True,
        center_mode=center_mode,
        gated_features=[name for name in GROUP_C_FEATURES],
    )
    return model, report


def _window_index_of(raw: pd.DataFrame) -> np.ndarray:
    """Recompute each raw row's window index from the §10 windowing params."""
    raw = raw.reset_index(drop=True)
    from src.features.windowing import windowing_params

    window, stride = windowing_params()
    out = np.full(len(raw), -1, dtype=int)
    for (event, node), g in raw.groupby(["event_id", "node_id"], sort=False):
        n = len(g)
        w = 0
        start = 0
        idx = g.index.to_numpy()
        while start + window <= n:
            out[idx[start : start + window]] = w
            w += 1
            start += stride
    return out
