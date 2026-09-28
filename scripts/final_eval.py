#!/usr/bin/env python3
"""The sole one-use reader of ``data/heldout_locked/`` (Phase 8).

The lock is incremented before opening the corpus. A failed evaluation still
burns the one-use set; reruns need the explicit ``--i-accept-burning-the-test-set``
override, which permanently records the additional burn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOCK_PATH = ROOT / "reports/test_lock.json"
TEST_DIR = ROOT / "data/heldout_locked"
CLASSES = ("NORMAL", "WARNING", "CRITICAL")


def claim_eval(lock_path: Path, accept_burning: bool) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    count = int(lock.get("evals_run", 0))
    if count >= 1 and not accept_burning:
        raise SystemExit("locked test already evaluated; pass --i-accept-burning-the-test-set to record another burn")
    lock["evals_run"] = count + 1
    if accept_burning:
        lock["burn_override_used"] = True
    tmp = lock_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(lock_path)
    return lock


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_lock(lock: dict) -> None:
    files = lock["files"]
    actual = {name: _file_hash(TEST_DIR / name) for name in sorted(files)}
    if actual != files:
        raise RuntimeError(f"locked corpus sha256 mismatch: expected {files}, got {actual}")
    combined = hashlib.sha256("\n".join(f"{name}:{actual[name]}" for name in sorted(actual)).encode()).hexdigest()
    if combined != lock["sha256"]:
        raise RuntimeError("combined heldout corpus digest does not match reports/test_lock.json")


def _aligned_prob(model, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), len(CLASSES)), dtype=float)
    for j, label in enumerate(model.classes_):
        out[:, int(label)] = raw[:, j]
    return out


def _calibrate(prob: np.ndarray, calibration: dict) -> np.ndarray:
    out = np.column_stack([calibration["models"][c].predict(prob[:, i]) for i, c in enumerate(CLASSES)])
    out = np.clip(out, 1e-8, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _threshold_predictions(prob: np.ndarray, thresholds: dict) -> np.ndarray:
    pred = np.asarray(CLASSES)[prob.argmax(axis=1)].copy()
    ci, wi = CLASSES.index("CRITICAL"), CLASSES.index("WARNING")
    pred[prob[:, ci] >= float(thresholds["critical"])] = "CRITICAL"
    pred[(pred != "CRITICAL") & (prob[:, wi] >= float(thresholds["warning"]))] = "WARNING"
    return pred


def _ece(prob: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    confidence = prob.max(axis=1)
    correct = np.asarray(CLASSES)[prob.argmax(axis=1)] == y
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (confidence >= edges[i]) & (confidence < edges[i + 1] if i < bins - 1 else confidence <= edges[i + 1])
        if mask.any():
            total += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return total


def _class_metrics(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    one_hot = np.column_stack([y == label for label in CLASSES]).astype(int)
    return {
        "per_class": classification_report(y, pred, labels=list(CLASSES), output_dict=True, zero_division=0),
        "confusion_matrix_labels_normal_warning_critical": confusion_matrix(y, pred, labels=list(CLASSES)).tolist(),
        "pr_auc_macro_ovr": float(np.mean([average_precision_score(one_hot[:, i], prob[:, i]) for i in range(len(CLASSES))])),
        "f1_macro": float(f1_score(y, pred, labels=list(CLASSES), average="macro", zero_division=0)),
        "recall_critical": float(recall_score(y, pred, labels=["CRITICAL"], average=None, zero_division=0)[0]),
        "false_alarm_rate_normal": float(np.mean(pred[y == "NORMAL"] != "NORMAL")) if (y == "NORMAL").any() else None,
        "multiclass_brier": float(np.mean(np.sum((prob - one_hot) ** 2, axis=1))),
        "ece_top_label_15_bins": _ece(prob, y),
    }


def _alerts(frame: pd.DataFrame, prob: np.ndarray) -> dict:
    from src.risk.alert_engine import AlertEngine

    ordered = frame.reset_index(drop=True).copy()
    ordered["row_index"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["event_id", "node_id", "window_index"], kind="stable")
    engine = AlertEngine()
    state_by_row: dict[int, str] = {}
    for row in ordered.itertuples(index=False):
        p = prob[int(row.row_index)]
        level = engine.update(
            f"{row.event_id}/{row.node_id}",
            dict(zip(CLASSES, p, strict=True)),
            conditions={
                "spatial_coherence_above_threshold": bool(np.isfinite(getattr(row, "spatial_coherence", np.nan)) and row.spatial_coherence > 0.5),
                "displacement_trend_positive": bool(np.isfinite(row.velocity) and row.velocity > 0),
                "physics_residual_low": bool(np.isfinite(row.physics_residual) and abs(row.physics_residual) <= 1.0),
                "neighbour_confirmations": 0,
            },
        )
        state_by_row[int(row.row_index)] = level
    states = np.asarray([state_by_row[i] for i in range(len(frame))])
    results = pd.DataFrame({"event_id": frame.event_id.to_numpy(), "risk_label": frame.risk_label.to_numpy(),
                            "timestamp": frame.window_timestamp.to_numpy(), "alert_state": states})
    normal = results[results.risk_label == "NORMAL"]
    per_event_duration = results.groupby("event_id").timestamp.agg(lambda s: max(0.0, float(s.max() - s.min())) / 24.0)
    normal_days = float(per_event_duration.loc[normal.event_id.unique()].sum()) if len(normal) else 0.0
    false_event_episodes = 0
    for _, group in normal.groupby("event_id", sort=False):
        active = group.alert_state.isin(("WATCH", "WARNING", "CRITICAL")).to_numpy()
        false_event_episodes += int(np.sum(active & ~np.r_[False, active[:-1]]))
    event_times = results.groupby("event_id", sort=False)
    lead = []
    for _event, g in event_times:
        onset = g.loc[g.risk_label.isin(("WARNING", "CRITICAL")), "timestamp"]
        alerted = g.loc[g.alert_state.isin(("WATCH", "WARNING", "CRITICAL")), "timestamp"]
        if len(onset) and len(alerted):
            lead.append(float(onset.min() - alerted.min()))
    counts = results.alert_state.value_counts(normalize=True).to_dict()
    return {
        "false_alarms_per_normal_event_day": false_event_episodes / normal_days if normal_days else None,
        "normal_event_days": normal_days,
        "false_alert_episodes_on_normal_events": false_event_episodes,
        "median_lead_time_hours": float(np.median(lead)) if lead else None,
        "p10_lead_time_hours": float(np.quantile(lead, 0.1)) if lead else None,
        "lead_time_events": len(lead),
        "time_in_state_fraction": {str(k): float(v) for k, v in counts.items()},
        "neighbour_confirmations_available": False,
        "de_escalation": "not implemented in current AlertEngine; cannot be evaluated",
        "watch_probability_mapping": "P(WATCH or higher) = 1 - P(NORMAL)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-accept-burning-the-test-set", action="store_true")
    args = parser.parse_args()
    lock = claim_eval(LOCK_PATH, args.i_accept_burning_the_test_set)
    _verify_lock(lock)

    from src.features.build_feature_store import build_feature_store
    from src.risk.artifacts import load_model_artifact
    raw_path = TEST_DIR / "synthetic_nodes.csv"
    usecols = ["event_id", "node_id", "timestamp", "x", "y", "tilt_x", "tilt_y", "tilt_magnitude",
               "displacement", "strain", "vibration_rms", "vibration_peak", "battery", "RSSI", "SNR",
               "packet_loss", "anomaly_label", "risk_label", "progression_label", "fault_label"]
    raw = pd.read_csv(raw_path, usecols=usecols)
    coords = raw.groupby("node_id", sort=False)[["x", "y"]].first().reset_index()
    features, report = build_feature_store(raw, coords, center_mode="detected")
    if report.center_mode != "detected":
        raise RuntimeError("test feature build mode differs from the frozen train/serve mode")
    features = features.dropna(subset=["risk_label", "anomaly_label"]).reset_index(drop=True)
    y = features.risk_label.astype(str).to_numpy()
    y_anomaly = features.anomaly_label.to_numpy(dtype=int)
    params = yaml.safe_load((ROOT / "configs/model_params.yaml").read_text(encoding="utf-8"))
    model_dir = ROOT / "models/tuned"
    tuned = load_model_artifact(model_dir / "risk_xgboost_tuned.joblib", expected_schema_version="v2")
    default = load_model_artifact(model_dir / "risk_xgboost_default.joblib", expected_schema_version="v2")
    logistic = load_model_artifact(model_dir / "risk_logistic_baseline.joblib", expected_schema_version="v2")
    tuned_prob_raw = _aligned_prob(tuned["model"], tuned["transform"](features).to_numpy(dtype=float))
    calibration = tuned["extras"].get("calibration")
    tuned_prob = _calibrate(tuned_prob_raw, calibration) if calibration else tuned_prob_raw
    tuned_pred = _threshold_predictions(tuned_prob, params["risk_classifier"]["thresholds"])

    base_prob = _aligned_prob(default["model"], default["transform"](features).to_numpy(dtype=float))
    base_pred = np.asarray(CLASSES)[base_prob.argmax(axis=1)]
    logit_prob_raw = logistic["model"].predict_proba(logistic["transform"](features).to_numpy(dtype=float))
    logit_prob = np.zeros((len(features), len(CLASSES)))
    for j, c in enumerate(logistic["model"].classes_):
        logit_prob[:, CLASSES.index(str(c))] = logit_prob_raw[:, j]
    logit_pred = np.asarray(CLASSES)[logit_prob.argmax(axis=1)]
    from src.risk.baselines import threshold_rule_predict
    rule_cfg = params["decision_rule"]
    rule_pred = threshold_rule_predict(features, **rule_cfg)
    rule_prob = np.column_stack([rule_pred == c for c in CLASSES]).astype(float)

    result = {
        "evaluation": {"kind": "single_use_locked_final", "synthetic_corpus_only": True,
                       "dataset_version": lock["dataset_version"], "feature_schema_version": lock["feature_schema_version"],
                       "seed": lock["seed"], "sha256": lock["sha256"], "evals_run": lock["evals_run"],
                       "events": int(features.event_id.nunique()), "windows": len(features),
                       "center_mode": report.center_mode, "test_touched_once": lock["evals_run"] == 1},
        "risk_models": {
            "tuned_xgboost": _class_metrics(y, tuned_prob, tuned_pred),
            "default_xgboost": _class_metrics(y, base_prob, base_pred),
            "logistic_regression": _class_metrics(y, logit_prob, logit_pred),
            "threshold_rule": _class_metrics(y, rule_prob, rule_pred),
        },
        "anomaly_models": {},
        "alert_engine": _alerts(features, tuned_prob),
    }
    for name, artifact_name in (("default", "iforest_default"), ("tuned", "iforest_tuned")):
        artifact = load_model_artifact(model_dir / f"{artifact_name}.joblib", expected_schema_version="v2")
        x = artifact["transform"](features).to_numpy(dtype=float)
        score = -artifact["model"].score_samples(x)
        threshold = float(params["baseline_isolation_forest"]["score_threshold"] if name == "default"
                          else params["isolation_forest"]["score_threshold"])
        binary = score > threshold
        result["anomaly_models"][name] = {
            "pr_auc_anomaly": float(average_precision_score(y_anomaly, score)),
            "roc_auc_anomaly": float(roc_auc_score(y_anomaly, score)),
            "recall_at_development_healthy_threshold": float(np.mean(binary[y_anomaly == 1])) if (y_anomaly == 1).any() else None,
            "false_positive_rate_normal_anomaly_label": float(np.mean(binary[y_anomaly == 0])) if (y_anomaly == 0).any() else None,
            "threshold": threshold,
        }

    report_path = ROOT / "reports/final_eval.json"
    report_path.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    summary = ["# One-time locked evaluation", "", "All results are from the synthetic held-out corpus.", "",
               f"Lock sha256: `{lock['sha256']}`; seed `{lock['seed']}`; evaluations recorded: `{lock['evals_run']}`.", "",
               "## Risk models", "", "| Model | Macro PR-AUC | Macro F1 | CRITICAL recall | NORMAL false-alarm rate |", "|---|---:|---:|---:|---:|"]
    for name, metrics in result["risk_models"].items():
        summary.append(f"| {name} | {metrics['pr_auc_macro_ovr']:.4f} | {metrics['f1_macro']:.4f} | {metrics['recall_critical']:.4f} | {metrics['false_alarm_rate_normal']:.4f} |")
    summary += ["", "## Anomaly models", "", "| Model | PR-AUC | Recall at development threshold | False-positive rate |", "|---|---:|---:|---:|"]
    for name, metrics in result["anomaly_models"].items():
        summary.append(f"| {name} | {metrics['pr_auc_anomaly']:.4f} | {metrics['recall_at_development_healthy_threshold']:.4f} | {metrics['false_positive_rate_normal_anomaly_label']:.4f} |")
    summary += ["", "## Alert engine", "", json.dumps(result["alert_engine"], indent=2), "",
                "The corpus has one node per event, so neighbour confirmation is unavailable and CRITICAL escalation remains gated.",
                "The present alert engine has no de-escalation implementation; that behavior is not reported as measured.", ""]
    (ROOT / "reports/final_eval.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(result["evaluation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
