#!/usr/bin/env python3
"""T-070 — the mandatory §25 ablation study (A–F) → experiments/ablation_a_to_f.json.

§25 mandates a six-arm ablation as the evidence that gates whether temporal
DL (§26 G) and GNN (§27 H) are ever built. Each arm adds ONE modality family
to the previous arm and the §24 metric families are re-measured:

    A  sensors only          (Group A: tilt / displacement / strain)
    B  + temporal            (Group B, per-channel + bare)
    C  + spatial             (Group C)
    D  + vibration + health  (Groups D + E — remaining on-node sensor channels)
    E  + physics             (Group F: expected field + residuals)
    F  + Sentinel-1          (Group H: node-level LOS snapshot, T-062)

Honest deviations from the §25 letter list, recorded in the output metadata:

- **No DGPS arm.** §19 makes DGPS sparse (6 control points) and
  evaluation-target-only (T-063 `assert_evaluation_only`); it cannot be a
  model input at scale, so a "+DGPS" arm would be theatre. Documented, not
  silently dropped.
- **No G/H temporal-DL or GNN arms** — those are the very architectures the
  ablation gates (§26/§27); building them to evaluate them is circular.
- The Isolation-Forest `anomaly_score` cross-signal is held OUT of every arm
  (constant across arms ⇒ cancels in comparisons); `physics_residual` enters
  only in arm E, where Group F is the arm's delta.

All metrics come from `src/evaluation/metrics.py` (T-069, §24 — no bare
accuracy anywhere). Splits are the T-068 event-level splits. Deterministic:
config-driven hyperparameters, fixed seeds, no sampling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import f1_score as sk_f1

from src.evaluation.metrics import (
    brier_score,
    classification_metrics,
    expected_calibration_error,
    false_alarms_per_day,
    hotspot_localisation_error,
    iou_hotspots,
    lead_time_stats,
    regression_metrics,
)
from src.features.group_b_temporal import GROUP_B_FEATURES
from src.risk.xgboost_model import train_risk_model
from src.simulator.grid import build_grid

OUT_JSON = REPO_ROOT / "experiments" / "ablation_a_to_f.json"
STORE = REPO_ROOT / "data" / "features" / "features_v2.parquet"
SPLITS = REPO_ROOT / "data" / "features" / "split_assignment.csv"
INSAR_CSV = REPO_ROOT / "experiments" / "nb08_mesh_aligned_insar.csv"

ALERT_THRESHOLD = 0.5  # P(CRITICAL) at which the alert engine would fire
HOTSPOT_RADIUS_M = 75.0  # 3 × §10 node spacing — the §24 match radius
STRIDE_HOURS = 0.6  # §10 window stride (36 min)
EVENT_DAYS = 1.0  # every §10 event spans one simulated day (§10 scenarios)

A_PHYSICAL = ("tilt_x", "tilt_y", "tilt_magnitude", "displacement", "strain")
B_TEMPORAL = tuple(dict.fromkeys(
    [f"tilt_{ch}_{feat}" for ch in ("x", "y") for feat in GROUP_B_FEATURES] + list(GROUP_B_FEATURES)
))
C_SPATIAL = ("neighbor_mean", "neighbor_std", "neighbor_anomaly_fraction", "spatial_coherence",
             "local_gradient", "local_strain", "hotspot_density", "distance_to_subsidence_center")
D_VIBRATION = ("vibration_rms", "vibration_peak", "crest_factor", "band_energy_low",
               "band_energy_mid", "band_energy_high", "spectral_centroid")
E_HEALTH = ("battery", "RSSI", "SNR", "packet_loss", "missing_ratio",
            "stuck_sensor_flag", "drift_score")
F_PHYSICS = ("expected_displacement", "expected_tilt", "physics_residual", "physics_residual_velocity")
H_INSAR = ("LOS_displacement", "LOS_velocity", "LOS_acceleration", "cumulative_displacement",
           "coherence", "spatial_gradient", "local_hotspot_density")

ARMS: dict[str, list[str]] = {
    "A_sensors_only": [*A_PHYSICAL],
    "B_add_temporal": [*A_PHYSICAL, *B_TEMPORAL],
    "C_add_spatial": [*A_PHYSICAL, *B_TEMPORAL, *C_SPATIAL],
    "D_add_vibration_health": [*A_PHYSICAL, *B_TEMPORAL, *C_SPATIAL, *D_VIBRATION, *E_HEALTH],
    "E_add_physics": [*A_PHYSICAL, *B_TEMPORAL, *C_SPATIAL, *D_VIBRATION, *E_HEALTH, *F_PHYSICS],
    "F_add_sentinel1": [*A_PHYSICAL, *B_TEMPORAL, *C_SPATIAL, *D_VIBRATION, *E_HEALTH, *F_PHYSICS, *H_INSAR],
}
ARM_DEVELOPMENT = {
    "A_sensors_only": "Group A (§13)",
    "B_add_temporal": "A + Group B",
    "C_add_spatial": "B + Group C",
    "D_add_vibration_health": "C + Groups D + E",
    "E_add_physics": "D + Group F (physics modality)",
    "F_add_sentinel1": "E + Group H (Sentinel-1 modality, T-062 mesh-aligned LOS)",
}


def enrich_with_insar(store: pd.DataFrame) -> pd.DataFrame:
    """Join the T-062 mesh-aligned Sentinel-1 features onto the store.

    Honest-join note: the mesh-aligned product carries ONE acquisition (the
    nb08 as-of snapshot, 400 nodes), and the synthetic store's
    ``window_timestamp`` is event-local hours — there is no wall-clock date
    to align a per-window join to. Sentinel-1 therefore enters arm F as a
    STATIC per-node LOS context (what a deployment genuinely knows about
    each node from the latest acquisition at inference time), NOT as a
    per-window observation. This limitation is recorded in the payload's
    ``deviations_from_prd`` — a real temporal stack is T-057's missing
    scenes, not something to fake here.
    """
    insar = pd.read_csv(INSAR_CSV)
    keep = ["node_id", *H_INSAR]
    insar = insar[keep].drop_duplicates("node_id")
    merged = store.merge(insar, on="node_id", how="left", validate="many_to_one")
    return merged


def alert_frame(model, test: pd.DataFrame) -> pd.DataFrame:
    """Per-window alert decisions + scores on the test split."""
    proba = model.predict_proba(test)
    classes = list(model.classes)
    out = test[["event_id", "node_id", "window_index", "window_timestamp",
                "anomaly_label", "risk_label", "nx", "ny"]].copy()
    out["p_critical"] = proba[:, classes.index("CRITICAL")]
    out["pred_label"] = np.asarray(classes)[proba.argmax(axis=1)]
    out["alarm_score"] = 1.0 - proba[:, classes.index("NORMAL")]
    return out


def evaluate_arm(arm_id: str, features: list[str], store: pd.DataFrame) -> dict:
    model = train_risk_model(store, feature_groups=features)
    test = store[store["split"] == "test"].reset_index(drop=True)
    af = alert_frame(model, test)

    # ---- detection (§24) ----------------------------------------------------
    y_true = (af["anomaly_label"] > 0).astype(int).to_numpy()
    y_pred = (af["pred_label"] != "NORMAL").astype(int).to_numpy()
    det = classification_metrics(y_true, y_pred, y_score=af["alarm_score"].to_numpy())
    det["macro_f1_risk3"] = float(
        sk_f1(af["risk_label"], af["pred_label"], average="macro", labels=sorted(af["risk_label"].unique()))
    )

    # ---- calibration of P(CRITICAL) ----------------------------------------
    p_crit = af["p_critical"].to_numpy()
    two_col = np.stack([p_crit, 1.0 - p_crit], axis=1)  # class 0 = CRITICAL
    y_idx = (af["risk_label"] != "CRITICAL").astype(int).to_numpy()
    calib = {"brier": brier_score(two_col, y_idx),
             "ece": expected_calibration_error(two_col, y_idx)}

    # ---- deformation error (model-invariant; §24 honesty) -------------------
    defo = regression_metrics(test["displacement"], test["expected_displacement"])

    # ---- spatial family ------------------------------------------------------
    ious: list[float] = []
    true_pts: list[tuple[float, float]] = []
    pred_pts: list[tuple[float, float]] = []
    windows_any_hot = 0
    for (_ev, _w), g in af.groupby(["event_id", "window_index"], sort=False):
        true_hot = (g["anomaly_label"] > 0).to_numpy()
        pred_hot = (g["p_critical"] >= ALERT_THRESHOLD).to_numpy()
        if true_hot.any() or pred_hot.any():
            windows_any_hot += 1
            ious.append(iou_hotspots(pred_hot.astype(int), true_hot.astype(int)))
        if true_hot.any():
            true_pts.extend(zip(g.loc[true_hot, "nx"], g.loc[true_hot, "ny"], strict=True))
            if pred_hot.any():
                pred_pts.extend(zip(g.loc[pred_hot, "nx"], g.loc[pred_hot, "ny"], strict=True))
    spatial = {
        "mean_iou_hotspots": float(np.mean(ious)) if ious else float("nan"),
        "windows_any_hot": windows_any_hot,
        **hotspot_localisation_error(np.array(true_pts), np.array(pred_pts) if pred_pts else np.zeros((0, 2)),
                                     match_radius_m=HOTSPOT_RADIUS_M),
    }

    # ---- operational / temporal ----------------------------------------------
    alerts: dict[str, float] = {}
    onsets: dict[str, float] = {}
    for ev, g in af.groupby("event_id", sort=False):
        g = g.sort_values("window_index")
        fired = g.loc[g["p_critical"] >= ALERT_THRESHOLD, "window_index"]
        if len(fired):
            alerts[str(ev)] = float(fired.iloc[0])
        onset = g.loc[g["anomaly_label"] > 0, "window_index"]
        if len(onset):
            onsets[str(ev)] = float(onset.iloc[0])
    temporal = lead_time_stats(alerts, onsets, stride_hours=STRIDE_HOURS)
    temporal["false_alarms_per_day"] = false_alarms_per_day(
        y_true, y_pred, n_days=float(af["event_id"].nunique()) * EVENT_DAYS
    )

    return {
        "arm": arm_id,
        "development": ARM_DEVELOPMENT[arm_id],
        "n_features": len(features),
        "features": features,
        "insar_nan_fraction": float(np.mean(~np.isfinite(test[list(H_INSAR)].to_numpy(dtype=float)))) if any(
            f in H_INSAR for f in features) else None,
        "detection": det,
        "calibration": calib,
        "deformation_error": defo,
        "spatial": spatial,
        "temporal": temporal,
    }


def main() -> int:
    store = pd.read_parquet(STORE)
    splits = pd.read_csv(SPLITS)
    store = store.merge(splits[["event_id", "split"]], on="event_id", how="left", validate="many_to_one")
    if store["split"].isna().any():
        raise SystemExit("split_assignment missing events — refusing to ablate on an unsplit store")

    grid = build_grid()
    coord = pd.DataFrame({"node_id": grid.node_ids, "nx": grid.x, "ny": grid.y})
    store = store.merge(coord, on="node_id", how="left", validate="many_to_one")

    store = enrich_with_insar(store)
    missing_all = [f for f in ARMS["F_add_sentinel1"] if f not in store.columns]
    if missing_all:
        raise SystemExit(f"ablation feature columns missing from the enriched store: {missing_all}")

    arms = {}
    for arm_id, features in ARMS.items():
        print(f"=== arm {arm_id} ({len(features)} features) ===", flush=True)
        arms[arm_id] = evaluate_arm(arm_id, features, store)
        a = arms[arm_id]
        print(f"  pr_auc={a['detection']['pr_auc']:.3f} f1={a['detection']['f1']:.3f} "
              f"brier={a['calibration']['brier']:.3f} "
              f"median_lead={a['temporal']['median_lead_time_hours']:.2f}h "
              f"fa/day={a['temporal']['false_alarms_per_day']:.3f} "
              f"iou={a['spatial']['mean_iou_hotspots']:.3f}", flush=True)

    payload = {
        "metadata": {
            "task": "T-070 (§25 mandatory ablation A–F)",
            "store": str(STORE.relative_to(REPO_ROOT)),
            "n_store_rows": int(len(store)),
            "splits": {k: int(v) for k, v in store["split"].value_counts().items()},
            "alert_threshold_p_critical": ALERT_THRESHOLD,
            "hotspot_match_radius_m": HOTSPOT_RADIUS_M,
            "window_stride_hours": STRIDE_HOURS,
            "event_span_days": EVENT_DAYS,
            "deviations_from_prd": [
                "No +DGPS arm: §19 makes DGPS sparse and evaluation-target-only "
                "(T-063 assert_evaluation_only); it cannot be a model input at scale.",
                "No temporal-DL/GNN arms: §26/§27 architectures are what this ablation gates.",
                "Isolation-Forest anomaly_score held out of every arm (constant across arms).",
                "Sentinel-1 arm F joins the node-level LOS snapshot from the single "
                "available acquisition (T-062/nb08) as static per-node context — no "
                "per-window InSAR exists because the synthetic events carry no "
                "wall-clock dates.",
            ],
        },
        "arms": arms,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
