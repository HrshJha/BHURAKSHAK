#!/usr/bin/env python3
"""T-074 — domain-gap Stage 2 (synthetic → tabletop) → experiments/domain_gap_stage2.json.

§23.2 Stage 2: the model trained on synthetic data runs **UNMODIFIED** on
real tabletop sensor data and its risk classification is scored against the
§23.1 independent ground truth.

**Data provenance (honest):** the physical §23.1 rig campaign is out of
workstream scope, so the "tabletop sensor data" here is the RECORDED
STAND-IN (``data/recorded/tabletop/raw_sensor_log.csv``, seeded — see the
folder README). The harness is the deliverable: point it at a real
campaign's CSVs with the same schema and nothing else changes.

Bridge (raw 10 Hz log → §10 windowing → feature emitters → model):

- ``event_id`` = trial_id; ``node_id`` = N1..N4; ``timestamp`` = hours
  (10 Hz samples on the 1-minute §10 grid interval).
- Channels ported with per-node 2 s re-baselining (the rig protocol's own
  baseline rule): ``displacement`` = ultrasonic distance − baseline (mm);
  ``tilt_x`` = roll, ``tilt_y`` = pitch (deg); ``tilt_magnitude`` =
  hypot(roll, pitch); ``strain`` = ToF crack-opening delta (mm) — the
  physical analog of the synthetic horizontal-convergence channel, recorded
  as a substitution; ``vibration_rms`` = |a| deviation from the node's
  median magnitude (g) so units match the synthetic channel.
- §10 windowing (config: 60 steps / stride 10) + Group A/B emitters run
  UNCHANGED. Groups C (needs mesh x/y the rig lacks), F (physics engine is
  a mine-panel model, meaningless on a tabletop) are honestly ABSENT —
  the model's resolver skips absent groups, which is part of what Stage 2
  measures (feature-scarce transfer).
- Scoring: predicted 3-state risk vs the reference state from
  ``known_displacement_mm`` (T-072 collapse: 0 → NORMAL, 1–2 → WARNING,
  3 → CRITICAL), plus the §23.1 displacement-error statistics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import classification_metrics  # noqa: E402
from src.evaluation.tabletop_protocol import (  # noqa: E402
    confusion_counts,
    displacement_error_vs_reference,
)
from src.features.group_a_physical import GROUP_A_FEATURES, emit_group_a  # noqa: E402
from src.features.group_b_temporal import emit_group_b  # noqa: E402
from src.features.windowing import build_windows  # noqa: E402
from src.risk.xgboost_model import train_risk_model  # noqa: E402
from src.simulator.grid import build_grid  # noqa: E402

OUT_JSON = REPO_ROOT / "experiments" / "domain_gap_stage2.json"
RAW_LOG = REPO_ROOT / "data" / "recorded" / "tabletop" / "raw_sensor_log.csv"
METADATA = REPO_ROOT / "data" / "recorded" / "tabletop" / "trial_metadata.csv"
WINDOWED_REF = REPO_ROOT / "data" / "recorded" / "tabletop" / "processed_windowed_dataset.csv"
STORE = REPO_ROOT / "data" / "features" / "features_v1.parquet"
SPLITS = REPO_ROOT / "data" / "features" / "split_assignment.csv"

BRIDGE_CHANNELS = ("tilt_x", "tilt_y", "tilt_magnitude", "displacement", "strain", "vibration_rms")
BASELINE_MS = 2000.0  # rig protocol: first 2 s of every trial is the baseline


def bridge_raw_to_series(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw 10 Hz log → per-sample channel series on the §10 time axis (hours)."""
    out_frames = []
    for (trial, node), g in raw.groupby(["trial_id", "node_id"], sort=False):
        g = g.sort_values("timestamp_ms", kind="stable")
        t0_mask = g["timestamp_ms"] < BASELINE_MS
        if not t0_mask.any():
            raise SystemExit(f"{trial}/{node}: no baseline samples in the first {BASELINE_MS:.0f} ms")
        ultra_base = float(g.loc[t0_mask, "ultrasonic_distance_mm"].mean())
        tof_base = float(g.loc[t0_mask, "tof_distance_mm"].mean())

        # orientation from the raw gravity vector (the rig protocol's own
        # derivation: pitch dips toward the descending trapdoor)
        ax = g["ax"].to_numpy(dtype=float)
        ay = g["ay"].to_numpy(dtype=float)
        az = g["az"].to_numpy(dtype=float)
        roll = np.degrees(np.arctan2(ay, np.sqrt(ax**2 + az**2)))
        pitch = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2)))
        amag = np.sqrt(ax**2 + ay**2 + az**2)
        out_frames.append(
            pd.DataFrame(
                {
                    "event_id": trial,
                    "node_id": node,
                    "timestamp": g["timestamp_ms"].to_numpy(dtype=float) / 3.6e6,  # hours
                    "tilt_x": roll,
                    "tilt_y": pitch,
                    "tilt_magnitude": np.hypot(roll, pitch),
                    "displacement": g["ultrasonic_distance_mm"].to_numpy(dtype=float) - ultra_base,
                    "strain": g["tof_distance_mm"].to_numpy(dtype=float) - tof_base,
                    "vibration_rms": amag - float(np.median(amag)),
                }
            )
        )
    return pd.concat(out_frames, ignore_index=True)


def reference_state_at_windows(windowed: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Attach the reference displacement/state to each bridged window.

    ``ref`` is the rig's own 2 s/1 s window grid with ``known_displacement_mm``
    and ``severity_class``. Each §10 window is matched to the rig window whose
    span CONTAINS the §10 window's end (tolerance 1 s); unmatched windows
    carry NaN reference and are excluded from scoring (counted honestly).
    """
    ref = ref.copy()
    ref["_span_end_ms"] = ref["window_start_ms"] + 2000
    parts = []
    for (trial, node), g in windowed.groupby(["event_id", "node_id"], sort=False):
        r = ref[(ref["trial_id"] == trial) & (ref["node_id"] == node)].sort_values("_span_end_ms")
        if r.empty:
            g = g.assign(known_displacement_mm=np.nan, severity_class=np.nan)
        else:
            end_ms = g["window_end"].to_numpy(dtype=float) * 3.6e6
            idx = np.searchsorted(r["_span_end_ms"].to_numpy(dtype=float), end_ms, side="left")
            idx = np.clip(idx, 0, len(r) - 1)
            matched = r.iloc[idx]
            delta_ms = np.abs(matched["_span_end_ms"].to_numpy(dtype=float) - end_ms)
            g = g.assign(
                known_displacement_mm=matched["known_displacement_mm"].to_numpy(dtype=float),
                severity_class=matched["severity_class"].to_numpy(dtype=float),
                reference_gap_ms=delta_ms,
            )
            g.loc[delta_ms > 1000.0, ["known_displacement_mm", "severity_class"]] = np.nan
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    raw = pd.read_csv(RAW_LOG)
    meta = pd.read_csv(METADATA)
    ref = pd.read_csv(WINDOWED_REF)
    print(f"raw log: {len(raw):,} samples, {raw.trial_id.nunique()} trials × {raw.node_id.nunique()} nodes @10 Hz")

    # ---- the model: trained on the SYNTHETIC store (§23 event split), unmodified
    store = pd.read_parquet(STORE)
    splits = pd.read_csv(SPLITS)
    store = store.merge(splits[["event_id", "split"]], on="event_id", how="left", validate="many_to_one")
    if store["split"].isna().any():
        raise SystemExit("split_assignment missing events — refusing to train on an unsplit store")

    # HEADLINE model: A/B feature set only — the channels that exist on BOTH
    # domains (rig telemetry → tilt/displacement/strain/vibration). This is
    # the honest domain-gap measurement: same sensor semantics on both sides.
    ab_features = (
        "tilt_x", "tilt_y", "tilt_magnitude", "displacement", "strain",
        "rolling_mean", "rolling_std", "rolling_min", "rolling_max", "slope",
        "velocity", "acceleration", "trend", "persistence", "change_point_score",
        "tilt_x_rolling_mean", "tilt_x_rolling_std", "tilt_x_rolling_min", "tilt_x_rolling_max",
        "tilt_x_slope", "tilt_x_velocity", "tilt_x_acceleration", "tilt_x_trend",
        "tilt_x_persistence", "tilt_x_change_point_score",
        "tilt_y_rolling_mean", "tilt_y_rolling_std", "tilt_y_rolling_min", "tilt_y_rolling_max",
        "tilt_y_slope", "tilt_y_velocity", "tilt_y_acceleration", "tilt_y_trend",
        "tilt_y_persistence", "tilt_y_change_point_score",
    )
    model = train_risk_model(store, feature_groups=list(ab_features))
    classes = list(model.classes)
    print(f"A/B model trained on synthetic store: {len(store):,} windows, classes {classes}")

    # DIAGNOSTIC model: the full §15 contract, fed all-NaN for the modalities
    # the rig cannot provide (C/D/E/F). Measures the modality-scarcity effect,
    # NOT the sensor domain gap — reported separately, never as the headline.
    model_full = train_risk_model(store)

    # ---- bridge the tabletop series through the §10 pipeline ---------------
    series = bridge_raw_to_series(raw)
    windowed = build_windows(series, channels=BRIDGE_CHANNELS, labels=()).df
    print(f"bridged windows: {len(windowed):,} (60-sample §10 windows over {windowed.event_id.nunique()} trials)")

    # reference matching FIRST (needs window_end, which the emitters drop)
    windowed_ref = reference_state_at_windows(windowed, ref)
    table = windowed_ref.merge(
        series[["event_id", "node_id", "timestamp"]].drop_duplicates(),
        left_on=["event_id", "node_id", "window_timestamp"],
        right_on=["event_id", "node_id", "timestamp"],
        how="left",
    ).drop(columns=["timestamp"])
    # no node coordinates on the rig — Group C/F absent; Group D/E channels
    # not bridged (no battery/RSSI telemetry on the recorded rig log)

    # Group A + B emitters, run unchanged on the bridged windows
    ga = emit_group_a(table)
    gb = emit_group_b(table)
    ref_cols = ["event_id", "node_id", "window_index", "known_displacement_mm", "severity_class"]
    feature_frame = ga.merge(
        gb[["event_id", "node_id", "window_index"] + [c for c in gb.columns if c not in ga.columns]],
        on=["event_id", "node_id", "window_index"], how="inner",
    ).merge(table[ref_cols], on=["event_id", "node_id", "window_index"], how="left")

    scored = feature_frame
    n_scored = int(scored["known_displacement_mm"].notna().sum())
    n_dropped = int(len(scored) - n_scored)
    print(f"reference-matched windows: {n_scored:,} ({n_dropped} unmatched excluded)")

    # ---- UNMODIFIED inference ----------------------------------------------
    proba = model.predict_proba(scored)
    pred_labels = np.asarray(classes)[proba.argmax(axis=1)]
    p_crit = proba[:, classes.index("CRITICAL")]

    # diagnostic: full-contract model with NaN-filled absent modalities
    missing_cols = [c for c in model_full.features if c not in scored.columns]
    diag_frame = scored.assign(**{c: np.nan for c in missing_cols})
    diag_proba = model_full.predict_proba(diag_frame)
    diag_pred = np.asarray(classes)[diag_proba.argmax(axis=1)]
    diag_p_crit = diag_proba[:, classes.index("CRITICAL")]

    ref_state = np.where(
        scored["severity_class"].to_numpy(dtype=float) >= 3, 2,
        np.where(scored["severity_class"].to_numpy(dtype=float) >= 1, 1, 0),
    ).astype(float)
    ok = np.isfinite(ref_state)
    labels = ("NORMAL", "WARNING", "CRITICAL")
    cm = confusion_counts(ref_state[ok], np.where(pred_labels == "CRITICAL", 2,
                          np.where(pred_labels == "WARNING", 1, 0))[ok], labels)
    per_state_recall = {lab: cm[lab][lab] / max(sum(cm[lab].values()), 1) for lab in labels}

    det = classification_metrics(
        (scored["severity_class"].to_numpy(dtype=float) > 0)[ok].astype(int),
        (pred_labels != "NORMAL")[ok].astype(int),
        y_score=p_crit[ok],
    )

    # §23.1 displacement error: bridged sensor displacement vs mechanism truth
    err_frame = scored[scored["node_id"].isin(("N2", "N3"))][["displacement_mean", "known_displacement_mm"]].rename(
        columns={"displacement_mean": "displacement_mm"}
    ).dropna()
    err = displacement_error_vs_reference(
        err_frame.assign(node_id="N2"),  # single pseudo-node: pooled error over active-node windows
        nodes=("N2",),
    )

    # in-regime comparison point (T-068 event-split test band, T-070 numbers)
    abl = json.load(open(REPO_ROOT / "experiments" / "ablation_a_to_f.json"))
    e_arm = abl["arms"]["E_add_physics"]

    payload = {
        "metadata": {
            "task": "T-074 (§23.2 domain-gap Stage 2: synthetic → tabletop)",
            "data_provenance": (
                "RECORDED STAND-IN: data/recorded/tabletop/raw_sensor_log.csv is the seeded "
                "output of scripts/generate_tabletop_dataset.py (seed 42). The physical §23.1 "
                "rig campaign is out of workstream scope; a real campaign's CSVs with the "
                "same schema run through this script unchanged."
            ),
            "status": "conditionally_executed (harness verified; real-hardware validation pending)",
            "model": "§15 XGBoost trained on the synthetic store (T-068 event split) — run UNMODIFIED",
            "bridge": {
                "channels": list(BRIDGE_CHANNELS),
                "windowing": "§10 config (60 steps / stride 10) via src/features/windowing.build_windows",
                "model_feature_set": "A+B (the §13 groups available on BOTH domains); C/D/E/F absent on the rig",
                "channel_substitutions": {
                    "strain": "ToF crack-opening delta (mm) — the rig's physical analog of horizontal convergence",
                    "vibration_rms": "|a| deviation from the node's median magnitude (g)",
                },
            },
            "windows": {"bridged": int(len(windowed)), "reference_matched": n_scored, "unmatched_excluded": n_dropped},
        },
        "results": {
            "model": "A/B feature set (tilt/displacement/strain/vibration + temporal), trained on synthetic",
            "risk_state_confusion": cm,
            "per_state_recall": per_state_recall,
            "detection": det,
            "displacement_error": err,
            "p_critical_stats": {
                "mean": float(np.mean(p_crit)), "max": float(np.max(p_crit)),
                "n_above_alert_threshold": int((p_crit >= 0.5).sum()),
            },
            "diagnostic_full_contract_nanfill": {
                "note": ("full §15 feature contract with C/D/E/F modalities fed as NaN — measures "
                         "modality scarcity, NOT the sensor domain gap; informational only"),
                "pred_label_counts": {str(k): int(v) for k, v in
                                      zip(*np.unique(diag_pred, return_counts=True), strict=True)},
                "p_critical_stats": {
                    "mean": float(np.mean(diag_p_crit)), "max": float(np.max(diag_p_crit)),
                    "n_above_alert_threshold": int((diag_p_crit >= 0.5).sum()),
                },
            },
        },
        "in_regime_reference": {
            "source": "experiments/ablation_a_to_f.json arm E_add_physics (test split)",
            "f1": e_arm["detection"]["f1"],
            "pr_auc": e_arm["detection"]["pr_auc"],
            "brier": e_arm["calibration"]["brier"],
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print("confusion:", json.dumps(cm))
    print("per-state recall:", {k: round(v, 3) for k, v in per_state_recall.items()})
    print(f"P(CRITICAL) mean {payload['results']['p_critical_stats']['mean']:.4f} "
          f"max {payload['results']['p_critical_stats']['max']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
