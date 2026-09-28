"""Development-only Data+ML pipeline — PRD §7 layered architecture (T-081).

Assembles the §7 layer cake IN ORDER, on development data, with no
stage bypassing the module that owns it:

1. **validation**   — `src/preprocessing.validation.validate_packets` on the
   raw §9 node records (DQ flags: missing / duplicate / out-of-order /
   corrupted).
2. **features**     — `src.features.build_feature_store.build_feature_store`
   (§10 windows → Groups A–F, §13 budget asserted).
3. **Isolation Forest** — `src.anomaly.isolation_forest.train_isolation_forest`
   on healthy-baseline TRAIN windows only; the score is joined back (§14→§15).
4. **spatial fusion** — Group C and §21.1 neighbor confirmations use only
   nodes co-temporal within the same event/window; current single-node events
   provide no spatial confirmation.
5. **physics check** — `src.physics.consistency.physics_engine` residuals for
   every window (Group F / §21), evaluated on the same mesh coordinates.
6. **XGBoost**      — `src.risk.xgboost_model.train_risk_model` (§15: groups
   A–F + `anomaly_score` + `physics_residual`), trained on TRAIN, early-stopped
   on VALIDATION.
7. **alert engine** — `src.risk.alert_engine.AlertEngine` over VALIDATION rows
   in window order (§21.1, config-driven, at most one escalation per update).
8. **explainability** — `src.risk.explainability.emit_risk_output` per scored
   VALIDATION window (FR-11: level + probabilities + contributing signals, never a
   bare probability).

§23 discipline: this runner uses only TRAIN and VALIDATION rows. The burned
legacy TEST split is never read for scoring; the locked corpus is reserved for
scripts/final_eval.py. The entry point is :func:`run_pipeline`, consumed by scripts/run_pipeline.py;
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


def _co_temporal_neighbor_confirmations(
    frame: pd.DataFrame, coords: pd.DataFrame, radius_m: float
) -> np.ndarray:
    """Count flagged neighbors only within the same event and window.

    Reused node IDs across independent events never confirm one another.
    """
    required = {"event_id", "node_id", "window_index", "if_flag"}
    if not required <= set(frame.columns):
        raise PipelineError(f"neighbor confirmation frame missing {sorted(required - set(frame.columns))}")
    frame = frame.reset_index(drop=True)
    xy = coords.set_index("node_id")[["x", "y"]]
    confirmations = np.zeros(len(frame), dtype=int)
    for (_event, _window), group in frame.groupby(["event_id", "window_index"], sort=False):
        group = group.sort_values("node_id", kind="stable")
        ids = group["node_id"].to_numpy()
        if len(ids) < 2:
            continue
        points = xy.loc[ids].to_numpy(dtype=float)
        distance = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1))
        np.fill_diagonal(distance, np.inf)
        flags = group["if_flag"].to_numpy(dtype=bool)
        counts = ((distance <= float(radius_m)) * flags[None, :]).sum(axis=1)
        confirmations[group.index.to_numpy()] = counts
    return confirmations


@dataclass
class PipelineResult:
    """Everything the §7 chain produced, with per-stage provenance."""

    validation: dict
    feature_report: object
    if_model: object
    risk_model: object
    scored_validation: pd.DataFrame    # per-window development output, never test
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
    """Execute the §7 chain on development data and score the validation rows."""
    from src.anomaly.isolation_forest import train_isolation_forest
    from src.features.build_feature_store import build_feature_store
    from src.preprocessing.validation import validate_packets
    from src.risk.alert_engine import AlertEngine
    from src.risk.explainability import emit_risk_output
    from src.risk.xgboost_model import train_risk_model

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    repo = Path(__file__).resolve().parent.parent
    splits_path = repo / "data" / "features" / "split_assignment.csv"
    if not splits_path.is_file():
        raise PipelineError(
            f"{splits_path.relative_to(repo)} is missing — produce it first with "
            "`python scripts/make_split_assignment.py` (the split file's single producer)"
        )
    splits = pd.read_csv(splits_path)

    vr = validate_packets(raw)
    vsum = vr.summary()
    log(f"[1/8] validation: {len(vr.df):,} rows → {vsum['flagged_rows']:,} DQ-flagged "
        f"{ {k: v for k, v in vsum.items() if k != 'flagged_rows'} }")

    model, report = build_feature_store(raw, coords, center_mode="detected")
    log(f"[2/8] features: {report.n_windows:,} windows, {len(report.features)} features (§13 budget OK)")

    # §23 split at event level (T-068) — read at stage 0, merged before any fitting
    model = model.merge(splits[["event_id", "split"]], on="event_id", how="left", validate="many_to_one")
    if model["split"].isna().any():
        raise PipelineError("events missing from split_assignment.csv — refusing an unsplit run")
    model = model.dropna(subset=["risk_label"])

    # Isolation Forest (§14, healthy-baseline TRAIN only)
    if_model = train_isolation_forest(model)
    model["anomaly_score"] = if_model.anomaly_score(model)
    model["if_flag"] = if_model.flags(model)
    log(f"[3/8] isolation forest: {len(if_model.features)} features, "
        f"threshold {if_model.threshold:.4f} ({if_model.threshold_rule})")

    # A confirmation is valid only among nodes observed in the same event and
    # window. Reused node IDs in unrelated single-node events never confirm.
    from src.features.group_c_spatial import neighbour_radius_m

    radius = float(neighbour_radius_m())
    model["neighbour_confirmations"] = _co_temporal_neighbor_confirmations(model, coords, radius)
    log(f"[4/8] spatial fusion: co-temporal graph radius {radius:g} m (config); "
        f"median confirmations {int(np.median(model['neighbour_confirmations']))}; "
        "single-node events cannot satisfy §21.1 neighbor confirmation")

    model = _physics_residuals(model, coords)
    # §21/§22 semantics: "physics_residual_low" is judged on the NORMALISED
    # residual (|z| <= 1.0, RESIDUAL_Z_OK in src/risk/explainability.py) —
    # standardised on TRAIN only; no held-out test rows are loaded here.
    tr = model["split"] == "train"
    mu = float(model.loc[tr, "physics_residual"].mean())
    sd = float(model.loc[tr, "physics_residual"].std())
    if not np.isfinite(sd) or sd <= 0:
        raise PipelineError("physics_residual has zero variance on the train split")
    model["physics_residual_z"] = (model["physics_residual"] - mu) / sd
    log("[5/8] physics check: §21 residuals computed for every window "
        f"(z-standardised on train: mu={mu:.3f}, sd={sd:.3f})")

    risk_model = train_risk_model(model)
    log(f"[6/8] xgboost: {len(risk_model.features)} §15 inputs, classes {risk_model.classes}")

    validation = model[model["split"] == "validation"].sort_values(
        ["window_timestamp", "event_id", "node_id"], kind="stable"
    ).reset_index(drop=True)
    proba = risk_model.predict_proba(validation)
    classes = list(risk_model.classes)
    for i, c in enumerate(classes):
        validation[f"p_{c}"] = proba[:, i]
    validation["predicted_level"] = np.asarray(classes)[np.argmax(proba, axis=1)]

    engine = AlertEngine()
    levels: list[str] = []
    for _, row in validation.iterrows():
        p = {c: float(row[f"p_{c}"]) for c in classes}
        conditions = {
            "spatial_coherence_above_threshold": bool(row.get("spatial_coherence", 0) > 0),
            "displacement_trend_positive": bool(row.get("velocity", 0) > 0),  # bare §13 B-group name = displacement view
            "physics_residual_low": bool(abs(float(row["physics_residual_z"])) <= 1.0),
            "neighbour_confirmations": int(row["neighbour_confirmations"]),  # CONDITION_KEYS value, not the config name
        }
        state_key = f"{row['event_id']}:{row['node_id']}"
        levels.append(engine.update(state_key, p, conditions=conditions))
    validation["alert_level"] = levels
    timeline = pd.DataFrame(
        {"event_id": validation["event_id"], "node_id": validation["node_id"], "window_index": validation["window_index"], "alert_level": levels}
    )
    log(f"[7/8] validation-only alert-engine trace: {timeline.alert_level.value_counts().to_dict()}")

    # explainability (FR-11/§22, never a bare probability)
    # FR-14: every prediction is logged with the five traceability fields,
    # copied from the REGISTERED model entry (never hand-assembled). The
    # dataset/schema versions come from the §10.1 manifest itself.
    import hashlib
    import json

    from src.config import risk_model_config
    from src.risk.model_registry import ModelRegistry

    manifest = json.loads((repo / "data" / "synthetic" / "dataset_manifest.json").read_text())
    provenance_hash = hashlib.sha256((repo / "configs" / "feature_provenance.yaml").read_bytes()).hexdigest()
    risk_cfg = risk_model_config()["xgboost"]
    registry = ModelRegistry()
    registry.register_model(
        model_name="subsense_xgboost_risk",
        model_version="1.0.0",
        feature_version=str(manifest["feature_schema_version"]),
        training_dataset_version=str(manifest["dataset_version"]),
        split_name="train; validation used for early stopping",
        seed=int(risk_cfg["random_state"]),
        provenance_hash=provenance_hash,
        artifact_path=None,  # in-process model; artifact dump is the runner's choice
    )
    explanations = []
    for _, row in validation.iterrows():
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
        scored_validation=validation,
        alert_timeline=timeline,
        explanations=explanations,
    )
