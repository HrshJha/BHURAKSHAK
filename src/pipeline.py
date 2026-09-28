"""End-to-end Data+ML pipeline — PRD §7 layered architecture (T-081).

Assembles the §7 layer cake IN ORDER, on held-out (test-split) data, with no
stage bypassing the module that owns it:

1. **validation**   — `src/preprocessing.validation.validate_packets` on the
   raw §9 node records (DQ flags: missing / duplicate / out-of-order /
   corrupted).
2. **features**     — `src.features.build_feature_store.build_feature_store`
   (§10 windows → Groups A–F, §13 budget asserted).
3. **Isolation Forest** — `src.anomaly.isolation_forest.train_isolation_forest`
   on healthy-baseline TRAIN windows only; the score is joined back (§14→§15).
4. **spatial fusion** — Group C is fused inside the feature store (§13/§20;
   the G-5 `center_mode` is recorded per row); the same node graph supplies
   the §21.1 `neighbour_confirmation` counts.
5. **physics check** — `src.physics.consistency.physics_engine` residuals for
   every window (Group F / §21), evaluated on the same mesh coordinates.
6. **XGBoost**      — `src.risk.xgboost_model.train_risk_model` (§15: groups
   A–F + `anomaly_score` + `physics_residual`), trained on TRAIN, early-stopped
   on VALIDATION.
7. **alert engine** — `src.risk.alert_engine.AlertEngine` over the TEST split
   in window order (§21.1, config-driven, at most one escalation per update).
8. **explainability** — `src.risk.explainability.emit_risk_output` per scored
   TEST window (FR-11: level + probabilities + contributing signals, never a
   bare probability).

§23 discipline: TRAIN/VALIDATION are used for fitting and thresholding only;
the TEST split is scored exactly once and never influences any fitted value.
The entry point is :func:`run_pipeline`, consumed by scripts/run_pipeline.py;
every stage's provenance (module + function) is recorded in the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "PipelineError",
    "PipelineResult",
    "load_synthetic_corpus",
    "run_pipeline",
]

_STAGE_PROVENANCE = {
    "validation": "src.preprocessing.validation.validate_packets",
    "features": "src.features.build_feature_store.build_feature_store",
    "isolation_forest": "src.anomaly.isolation_forest.train_isolation_forest",
    "spatial_fusion": "src.features.group_c_spatial.emit_group_c (fused in build_feature_store)",
    "physics_check": "src.physics.consistency.physics_engine",
    "xgboost": "src.risk.xgboost_model.train_risk_model",
    "alert_engine": "src.risk.alert_engine.AlertEngine.update",
    "explainability": "src.risk.explainability.emit_risk_output",
}


class PipelineError(RuntimeError):
    """Raised when a pipeline stage's preconditions fail (fail loudly)."""


@dataclass
class PipelineResult:
    """Everything the §7 chain produced, with per-stage provenance."""

    validation: dict
    feature_report: object
    if_model: object
    risk_model: object
    scored_test: pd.DataFrame          # per-window: keys, proba, predicted level, alert level
    alert_timeline: pd.DataFrame       # per (node, window): §21.1 level after each update
    explanations: list                 # RiskExplanation objects (FR-11 payload)
    provenance: dict = field(default_factory=lambda: dict(_STAGE_PROVENANCE))


def load_synthetic_corpus(nodes_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the §12 synthetic node corpus + the T-068 event split assignment."""
    repo = Path(__file__).resolve().parent.parent
    nodes_path = nodes_path or repo / "data" / "synthetic" / "synthetic_nodes.csv"
    raw = pd.read_csv(
        nodes_path,
        usecols=[
            "event_id", "node_id", "timestamp", "x", "y",
            "tilt_x", "tilt_y", "tilt_magnitude", "displacement", "strain",
            "vibration_rms", "vibration_peak", "battery", "RSSI", "SNR", "packet_loss",
            "anomaly_label", "risk_label", "progression_label", "fault_label",
        ],
    )
    coords = raw.groupby("node_id", sort=False)[["x", "y"]].first().reset_index()
    return raw, coords


def _physics_residuals(model: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Stage 5 — §21 physics residuals on the same mesh coordinates (Group F)."""
    from src.physics.consistency import physics_engine

    phys = physics_engine()
    node_xy = model[["node_id"]].merge(coords, on="node_id", how="left", validate="many_to_one")
    if node_xy[["x", "y"]].isna().any().any():
        raise PipelineError("feature-store rows reference nodes outside the mesh coordinates")
    t_hours = model["window_timestamp"].to_numpy(dtype=float)
    model = model.copy()
    model["physics_residual"] = phys.displacement_residual(
        model["displacement"].to_numpy(dtype=float),
        node_xy["x"].to_numpy(dtype=float),
        node_xy["y"].to_numpy(dtype=float),
        t_hours,
    )
    etx, ety = phys.expected_tilt(
        node_xy["x"].to_numpy(dtype=float), node_xy["y"].to_numpy(dtype=float), t_hours
    )
    model["physics_residual_tilt_x"] = model["tilt_x"].to_numpy(dtype=float) - etx
    model["physics_residual_tilt_y"] = model["tilt_y"].to_numpy(dtype=float) - ety
    return model


def run_pipeline(raw: pd.DataFrame, coords: pd.DataFrame, *, verbose: bool = True) -> PipelineResult:
    """Execute the §7 chain end-to-end and score the held-out TEST split."""
    from src.anomaly.isolation_forest import train_isolation_forest
    from src.features.build_feature_store import build_feature_store
    from src.preprocessing.validation import validate_packets
    from src.risk.alert_engine import AlertEngine
    from src.risk.explainability import emit_risk_output
    from src.risk.xgboost_model import train_risk_model
    from src.simulator.grid import build_grid

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    # ---- 0. §23 split file must exist BEFORE any compute (fail fast) -------
    repo = Path(__file__).resolve().parent.parent
    splits_path = repo / "data" / "features" / "split_assignment.csv"
    if not splits_path.is_file():
        raise PipelineError(
            f"{splits_path.relative_to(repo)} is missing — produce it first with "
            "`python scripts/make_split_assignment.py` (the split file's single producer)"
        )
    splits = pd.read_csv(splits_path)

    # ---- 1. validation (§9/§11) -------------------------------------------
    vr = validate_packets(raw)
    vsum = vr.summary()
    log(f"[1/8] validation: {len(vr.df):,} rows → {vsum['flagged_rows']:,} DQ-flagged "
        f"{ {k: v for k, v in vsum.items() if k != 'flagged_rows'} }")

    # ---- 2. features (§10 windows → Groups A–F, §13 budget) ----------------
    model, report = build_feature_store(raw, coords, center_mode="oracle")
    log(f"[2/8] features: {report.n_windows:,} windows, {len(report.features)} features (§13 budget OK)")

    # §23 split at event level (T-068) — read at stage 0, merged before any fitting
    model = model.merge(splits[["event_id", "split"]], on="event_id", how="left", validate="many_to_one")
    if model["split"].isna().any():
        raise PipelineError("events missing from split_assignment.csv — refusing an unsplit run")
    model = model.dropna(subset=["risk_label"])

    # ---- 3. Isolation Forest (§14, healthy-baseline TRAIN only) ------------
    if_model = train_isolation_forest(model)
    model["anomaly_score"] = if_model.anomaly_score(model)
    model["if_flag"] = if_model.flags(model)
    log(f"[3/8] isolation forest: {len(if_model.features)} features, "
        f"threshold {if_model.threshold:.4f} ({if_model.threshold_rule})")

    # ---- 4. spatial fusion evidence (§20/§21.1) ----------------------------
    # Group C is already fused into the store; the §21.1 neighbour-confirmation
    # counts come from the SAME mesh graph and the SAME config-driven radius
    # the Group C fusion used (neighbour_radius_m = multiplier × spacing_m).
    from src.features.group_c_spatial import neighbour_radius_m

    grid = build_grid()
    coord_xy = pd.DataFrame({"node_id": grid.node_ids, "nx": grid.x, "ny": grid.y})
    xy = model[["node_id"]].drop_duplicates().merge(coord_xy, on="node_id", how="left", validate="many_to_one")
    if xy[["nx", "ny"]].isna().any().any():
        raise PipelineError("store nodes missing from the §10 mesh grid")
    nid = xy["node_id"].to_numpy()
    P = np.column_stack([xy["nx"].to_numpy(dtype=float), xy["ny"].to_numpy(dtype=float)])  # metres
    radius = float(neighbour_radius_m())
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    neighbours = {str(a): [str(b) for b in nid[(D[i] <= radius) & (nid != a)]] for i, a in enumerate(nid)}
    flags_by_node = model.groupby("node_id")["if_flag"].max().astype(bool).to_dict()
    model["neighbour_confirmations"] = [
        int(sum(bool(flags_by_node.get(nb, False)) for nb in neighbours.get(str(n), [])))
        for n in model["node_id"]
    ]
    log(f"[4/8] spatial fusion: Group C fused in-store; §21.1 graph "
        f"({len(neighbours)} nodes, radius {radius:g} m (config), "
        f"median neighbours {int(np.median([len(v) for v in neighbours.values()]))})")

    # ---- 5. physics check (§21) ---------------------------------------------
    model = _physics_residuals(model, coords)
    # §21/§22 semantics: "physics_residual_low" is judged on the NORMALISED
    # residual (|z| <= 1.0, RESIDUAL_Z_OK in src/risk/explainability.py) —
    # standardised on the TRAIN split only (§23: no test rows in the fit).
    tr = model["split"] == "train"
    mu = float(model.loc[tr, "physics_residual"].mean())
    sd = float(model.loc[tr, "physics_residual"].std())
    if not np.isfinite(sd) or sd <= 0:
        raise PipelineError("physics_residual has zero variance on the train split")
    model["physics_residual_z"] = (model["physics_residual"] - mu) / sd
    log("[5/8] physics check: §21 residuals computed for every window "
        f"(z-standardised on train: mu={mu:.3f}, sd={sd:.3f})")

    # ---- 6. XGBoost (§15) ----------------------------------------------------
    risk_model = train_risk_model(model)
    log(f"[6/8] xgboost: {len(risk_model.features)} §15 inputs, classes {risk_model.classes}")

    test = model[model["split"] == "test"].sort_values(
        ["window_timestamp", "event_id", "node_id"], kind="stable"
    ).reset_index(drop=True)
    proba = risk_model.predict_proba(test)
    classes = list(risk_model.classes)
    for i, c in enumerate(classes):
        test[f"p_{c}"] = proba[:, i]
    test["predicted_level"] = np.asarray(classes)[np.argmax(proba, axis=1)]

    # ---- 7. alert engine (§21.1, in window order) ---------------------------
    engine = AlertEngine()
    levels: list[str] = []
    for _, row in test.iterrows():
        p = {c: float(row[f"p_{c}"]) for c in classes}
        conditions = {
            "spatial_coherence_above_threshold": bool(row.get("spatial_coherence", 0) > 0),
            "displacement_trend_positive": bool(row.get("velocity", 0) > 0),  # bare §13 B-group name = displacement view
            "physics_residual_low": bool(abs(float(row["physics_residual_z"])) <= 1.0),
            "neighbour_confirmations": int(row["neighbour_confirmations"]),  # CONDITION_KEYS value, not the config name
        }
        levels.append(engine.update(str(row["node_id"]), p, conditions=conditions))
    test["alert_level"] = levels
    timeline = pd.DataFrame(
        {"node_id": test["node_id"], "window_index": test["window_index"], "alert_level": levels}
    )
    log(f"[7/8] alert engine: {timeline.alert_level.value_counts().to_dict()}")

    # ---- 8. explainability (FR-11/§22, never a bare probability) ------------
    # FR-14: every prediction is logged with the five traceability fields,
    # copied from the REGISTERED model entry (never hand-assembled). The
    # dataset/schema versions come from the §10.1 manifest itself.
    import json

    from src.risk.model_registry import ModelRegistry

    manifest = json.loads((repo / "data" / "synthetic" / "dataset_manifest.json").read_text())
    registry = ModelRegistry()
    registry.register_model(
        model_name="subsense_xgboost_risk",
        model_version="1.0.0",
        feature_version=str(manifest["feature_schema_version"]),
        training_dataset_version=str(manifest["dataset_version"]),
        artifact_path=None,  # in-process model; artifact dump is the runner's choice
    )
    explanations = []
    for _, row in test.iterrows():
        signals = {
            "displacement_trend": float(row.get("slope", 0.0)),  # bare §13 B-group name = displacement view
            "spatial_coherence": float(row.get("spatial_coherence", 0.0)),
            "anomaly_score": float(row["anomaly_score"]),
            "physics_residual": float(row["physics_residual_z"]),  # normalised (§22 z= text)
            "vibration_rms": float(row.get("vibration_rms", 0.0)),
        }
        _, _, expl = emit_risk_output(
            str(row["node_id"]), {c: float(row[f"p_{c}"]) for c in classes}, signals
        )
        explanations.append(expl)
        registry.log_prediction(
            "subsense_xgboost_risk",
            "1.0.0",
            {
                "node_id": str(row["node_id"]),
                "event_id": str(row["event_id"]),
                "window_index": int(row["window_index"]),
                "predicted_level": expl.predicted_level,
                "probabilities": dict(expl.probabilities),
                "alert_level": str(row["alert_level"]),
                "top_signal": expl.contributions[0].signal if expl.contributions else None,
            },
        )
    log(f"[8/8] explainability: {len(explanations):,} FR-11 payloads emitted, "
        f"FR-14 prediction log written ({len(explanations):,} records)")

    return PipelineResult(
        validation={"n_rows": int(len(vr.df)), **vsum},
        feature_report=report,
        if_model=if_model,
        risk_model=risk_model,
        scored_test=test,
        alert_timeline=timeline,
        explanations=explanations,
    )
