#!/usr/bin/env python3
"""Fit and register selected development models for evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import hashlib
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tune_models import CLASSES, development_frame
from src.anomaly.isolation_forest import healthy_baseline_mask
from src.config import anomaly_config, risk_model_config
from src.risk.artifacts import save_model_artifact
from src.risk.baselines import calibrate_threshold_rule


def _weight_vector(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(len(y), dtype=float)
    counts = {c: max(1, int(np.sum(y == c))) for c in CLASSES}
    base = {c: len(y) / (len(CLASSES) * counts[c]) for c in CLASSES}
    power = 1.0 if mode == "inverse" else 0.5
    return np.array([base[c] ** power for c in y], dtype=float)


def _fit_xgb(df: pd.DataFrame, features: list[str], params: dict, n_estimators: int, sample_weight: np.ndarray | None = None):
    y = df.risk_label.astype(str).to_numpy()
    numeric = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    clf = XGBClassifier(
        **params, n_estimators=int(n_estimators), objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", tree_method="hist", n_jobs=1, random_state=42,
    )
    clf.fit(df[features].to_numpy(dtype=float), numeric, sample_weight=sample_weight)
    return clf


def _aligned_proba(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    raw = model.predict_proba(frame[features].to_numpy(dtype=float))
    out = np.zeros((len(frame), len(CLASSES)))
    for j, class_id in enumerate(model.classes_):
        out[:, int(class_id)] = raw[:, j]
    return out


def _calibrate(raw: np.ndarray, y: np.ndarray, method: str) -> dict:
    models = {}
    for i, label in enumerate(CLASSES):
        target = (y == label).astype(int)
        if method == "isotonic":
            models[label] = IsotonicRegression(out_of_bounds="clip").fit(raw[:, i], target)
        else:
            model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            model.fit(raw[:, i].reshape(-1, 1), target)
            models[label] = model
    return {"method": method, "models": models}


def _apply_calibration(raw: np.ndarray, calibration: dict) -> np.ndarray:
    values = []
    for i, c in enumerate(CLASSES):
        model = calibration["models"][c]
        if calibration["method"] == "sigmoid":
            values.append(model.predict_proba(raw[:, i].reshape(-1, 1))[:, 1])
        else:
            values.append(model.predict(raw[:, i]))
    prob = np.column_stack(values)
    prob = np.clip(prob, 1e-8, 1.0)
    return prob / prob.sum(axis=1, keepdims=True)


def _ece(prob: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(CLASSES)[prob.argmax(axis=1)]
    conf = prob.max(axis=1)
    correct = pred == y
    value = 0.0
    edges = np.linspace(0, 1, 16)
    for i in range(15):
        mask = (conf >= edges[i]) & (conf < edges[i + 1] if i < 14 else conf <= edges[i + 1])
        if mask.any():
            value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(value)


def main() -> int:
    xgb_study = json.loads((ROOT / "reports/tuning/xgboost_study.json").read_text(encoding="utf-8"))
    if_study = json.loads((ROOT / "reports/tuning/isolation_forest_study.json").read_text(encoding="utf-8"))
    forecaster_path = ROOT / "reports/tuning/forecaster_study.json"
    if not forecaster_path.exists():
        raise SystemExit("forecaster Optuna study is required before freezing models")
    forecaster_study = json.loads(forecaster_path.read_text(encoding="utf-8"))
    frame, features = development_frame()
    train = frame[frame.split == "train"]
    val = frame[frame.split == "validation"]
    dev = frame[frame.split.isin(("train", "validation"))]
    manifest = json.loads((ROOT / "data/synthetic/dataset_manifest.json").read_text(encoding="utf-8"))
    dataset_version = str(manifest["dataset_version"])

    best_xgb = xgb_study.get("best")
    best_if = if_study.get("best")
    if not best_xgb or not best_if:
        raise SystemExit("both XGBoost and Isolation Forest studies need a completed trial")
    xgb_params = dict(best_xgb["params"])
    xgb_params.pop("class_weighting", None)
    xgb_params.pop("critical_threshold", None)
    xgb_params.pop("warning_threshold", None)
    weight_mode = str(best_xgb["params"]["class_weighting"])
    best_iterations = list(map(int, best_xgb.get("best_iterations", [])))
    if not best_iterations:
        from scripts.tune_models import grouped_folds
        for tr_ids, va_ids in grouped_folds(frame):
            fold_train = frame[frame.event_id.isin(tr_ids)]
            fold_val = frame[frame.event_id.isin(va_ids)]
            ytr = np.asarray([CLASSES.index(v) for v in fold_train.risk_label.astype(str)])
            yv = np.asarray([CLASSES.index(v) for v in fold_val.risk_label.astype(str)])
            weights = _weight_vector(fold_train.risk_label.astype(str).to_numpy(), weight_mode)
            probe = XGBClassifier(**xgb_params, n_estimators=2000, objective="multi:softprob", num_class=3,
                                  eval_metric="mlogloss", tree_method="hist", early_stopping_rounds=50,
                                  n_jobs=1, random_state=42)
            probe.fit(fold_train[features].to_numpy(dtype=float), ytr, sample_weight=weights,
                      eval_set=[(fold_val[features].to_numpy(dtype=float), yv)], verbose=False)
            best_iterations.append(int(getattr(probe, "best_iteration", 0)) + 1)
    tuned_estimators = max(1, int(np.median(best_iterations)))
    risk_model = _fit_xgb(train, features, xgb_params, tuned_estimators,
                          _weight_vector(train.risk_label.astype(str).to_numpy(), weight_mode))

    base_cfg = dict(risk_model_config()["xgboost"])
    base_cfg.pop("random_state", None)
    default_estimators = int(base_cfg.pop("n_estimators"))
    default_model = _fit_xgb(train, features, base_cfg, default_estimators)

    scaler = StandardScaler().fit(train[features].to_numpy(dtype=float))
    logistic_cfg = risk_model_config()["baselines"]["logistic"]
    logistic = LogisticRegression(C=float(logistic_cfg["C"]), max_iter=int(logistic_cfg["max_iter"]),
                                  class_weight="balanced", random_state=42)
    logistic.fit(scaler.transform(train[features].to_numpy(dtype=float)), train.risk_label.astype(str).to_numpy())

    val_event_y = val.groupby("event_id", sort=True).risk_label.agg(lambda s: s.mode().iloc[0])
    val_events = val_event_y.index.to_numpy()
    cal_idx, select_idx = next(StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42).split(
        val_events, val_event_y.to_numpy(), groups=val_events))
    calibration_frame = val[val.event_id.isin(val_events[cal_idx])]
    threshold_frame = val[val.event_id.isin(val_events[select_idx])]
    raw_cal = _aligned_proba(risk_model, calibration_frame, features)
    y_cal = calibration_frame.risk_label.astype(str).to_numpy()
    choices = {}
    for method in ("isotonic", "sigmoid"):
        candidate = _calibrate(raw_cal, y_cal, method)
        prob = _apply_calibration(_aligned_proba(risk_model, threshold_frame, features), candidate)
        ysel = threshold_frame.risk_label.astype(str).to_numpy()
        onehot = np.column_stack([ysel == c for c in CLASSES]).astype(float)
        brier = float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))
        ece = _ece(prob, ysel)
        choices[method] = {"model": candidate, "brier": brier, "ece": ece, "score": brier + ece}
    selected_method = min(choices, key=lambda k: choices[k]["score"])
    calibration = choices[selected_method]["model"]
    prob = _apply_calibration(_aligned_proba(risk_model, threshold_frame, features), calibration)
    ysel = threshold_frame.risk_label.astype(str).to_numpy()
    confidence = prob.max(axis=1)
    correct = np.asarray(CLASSES)[prob.argmax(axis=1)] == ysel
    edges = np.linspace(0.0, 1.0, 16)
    bin_conf, bin_acc = [], []
    for i in range(15):
        mask = (confidence >= edges[i]) & (confidence < edges[i + 1] if i < 14 else confidence <= edges[i + 1])
        if mask.any():
            bin_conf.append(float(confidence[mask].mean()))
            bin_acc.append(float(correct[mask].mean()))
    (ROOT / "reports/tuning").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", label="perfect calibration")
    ax.plot(bin_conf, bin_acc, marker="o", linewidth=1.8, label=f"{selected_method} (validation)")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted confidence", ylabel="Observed accuracy",
           title="Top-label reliability — development validation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/tuning/reliability.png", dpi=150)
    plt.close(fig)
    class_thresholds = {}
    operating = {}
    decision_targets = yaml.safe_load((ROOT / "configs/alerts.yaml").read_text(encoding="utf-8")).get("risk_decision_targets", {})
    for label in ("WARNING", "CRITICAL"):
        col = CLASSES.index(label)
        target = (ysel == label).astype(int)
        precision, recall, thresholds_curve = precision_recall_curve(target, prob[:, col])
        if not len(thresholds_curve):
            threshold = 0.5
            index = 0
            fpr = float(np.mean(prob[:, col] >= threshold))
            selected_recall, selected_precision = float(recall[0]), float(precision[0])
            target_met = False
        else:
            fpr_curve = np.array([np.mean((prob[:, col] >= t)[target == 0]) if np.any(target == 0) else 0.0
                                  for t in thresholds_curve])
            target_cfg = decision_targets.get(label, {})
            min_recall = float(target_cfg.get("minimum_recall", 0.8))
            max_fpr = float(target_cfg.get("maximum_false_positive_rate", 0.1))
            eligible = np.flatnonzero((recall[:-1] >= min_recall) & (fpr_curve <= max_fpr))
            if len(eligible):
                # Meet the declared recall and false-alarm point, then prefer
                # the candidate with the best validation precision.
                index = int(eligible[np.argmax(precision[:-1][eligible])])
                target_met = True
            else:
                # The operating point is unattainable on this validation
                # partition. Keep the false-alarm cap if possible and select
                # maximum recall; otherwise minimize FPR then maximize recall.
                within_cap = np.flatnonzero(fpr_curve <= max_fpr)
                pool = within_cap if len(within_cap) else np.arange(len(thresholds_curve))
                index = int(pool[np.lexsort((-recall[:-1][pool], fpr_curve[pool]))[0]])
                target_met = False
            threshold = float(thresholds_curve[index])
            fpr = float(fpr_curve[index])
            selected_recall, selected_precision = float(recall[index]), float(precision[index])
        class_thresholds[label.lower()] = threshold
        target_cfg = decision_targets.get(label, {})
        operating[label] = {"threshold": threshold, "precision": selected_precision,
                            "recall": selected_recall, "support": int(target.sum()),
                            "false_positive_rate": fpr, "target": {
                                "minimum_recall": target_cfg.get("minimum_recall"),
                                "maximum_false_positive_rate": target_cfg.get("maximum_false_positive_rate"),
                                "met": target_met,
                            },
                            "selection_reason": "highest validation precision meeting configured recall and false-alarm limits; if infeasible, best recall within false-alarm cap"}

    if_features = features
    healthy_train = train.loc[healthy_baseline_mask(train), if_features]
    healthy_val = val.loc[healthy_baseline_mask(val), if_features]
    if not len(healthy_train) or not len(healthy_val):
        raise SystemExit("healthy rows missing from train or validation; cannot set IF threshold")
    cfg_if = anomaly_config()
    default_if = IsolationForest(n_estimators=int(cfg_if["isolation_forest"]["n_estimators"]),
                                 contamination="auto", random_state=42, n_jobs=1)
    default_if.fit(healthy_train.to_numpy(dtype=float))
    tuned_if = IsolationForest(**best_if["params"], contamination="auto", random_state=42, n_jobs=1)
    tuned_if.fit(healthy_train.to_numpy(dtype=float))
    alpha = float(cfg_if["far_alpha"])
    thresholds = {
        "default": float(np.quantile(-default_if.score_samples(healthy_val.to_numpy(dtype=float)), alpha)),
        "tuned": float(np.quantile(-tuned_if.score_samples(healthy_val.to_numpy(dtype=float)), alpha)),
        "percentile": alpha,
    }
    params_yaml = {
        "forecaster": {
            "status": "tuned_on_development_groups" if forecaster_study.get("beats_persistence") else "search_complete_did_not_beat_persistence",
            **forecaster_study["best"]["params"],
            "channels": ["displacement", "tilt_x", "tilt_y"],
            "horizons": forecaster_study["study"]["horizons"],
            "seed": 42,
            "epochs": forecaster_study["study"]["epochs_per_trial"],
            "cv_mean_normalized_mse": forecaster_study["best"]["mean_normalized_mse"],
            "cv_mean_persistence_normalized_mse": forecaster_study["best"]["mean_persistence_normalized_mse"],
            "search_artifact": "reports/tuning/forecaster_study.json",
            "requested_trials": forecaster_study["study"]["requested_trials"],
            "actual_trials": forecaster_study["study"]["actual_trials"],
        },
        "risk_classifier": {
            "status": "frozen_after_development_grouped_cv",
            "params": best_xgb["params"], "features": features,
            "thresholds": {"critical": class_thresholds["critical"], "warning": class_thresholds["warning"]},
            "n_estimators": tuned_estimators,
            "calibration": {"method": selected_method,
                            "validation_metrics": {k: {"brier": v["brier"], "ece": v["ece"], "score": v["score"]}
                                                  for k, v in choices.items()},
                            "threshold_operating_points": operating},
            "objective_value": best_xgb["value"], "study": "reports/tuning/xgboost_study.json",
        },
        "isolation_forest": {
            "status": "frozen_after_development_grouped_cv",
            "params": best_if["params"], "features": if_features,
            "score_threshold": thresholds["tuned"], "threshold_percentile": alpha,
            "study": "reports/tuning/isolation_forest_study.json",
        },
        "baseline_isolation_forest": {"params": {"n_estimators": int(cfg_if["isolation_forest"]["n_estimators"]),
                                                    "contamination": "auto", "random_state": 42},
                                       "score_threshold": thresholds["default"]},
        "decision_rule": calibrate_threshold_rule(val)["params"],
        "alert_conditions": {
            "physics_residual_train_mean": float(train.physics_residual.mean()),
            "physics_residual_train_std": float(train.physics_residual.std(ddof=0)) or 1.0,
            "source_split": "train",
        },
        "training": {"dataset_version": dataset_version, "feature_schema_version": "v2",
                     "seed": 42, "groups": int(train.event_id.nunique()),
                     "split": "train; validation used only for calibration and thresholds", "test_touched": False},
    }
    (ROOT / "configs/model_params.yaml").write_text(yaml.safe_dump(params_yaml, sort_keys=False), encoding="utf-8")
    artifact_dir = ROOT / "models" / "tuned"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    common = {"training_dataset_version": dataset_version, "split_name": "train", "seed": 42}
    provenance_hash = hashlib.sha256((ROOT / "configs/feature_provenance.yaml").read_bytes()).hexdigest()
    save_model_artifact(artifact_dir / "risk_xgboost_tuned.joblib", model=risk_model, features=features,
                        model_name="BhuRakshak risk XGBoost tuned", model_version="v2.0.0-dev", **common,
                        extra={"class_names": list(CLASSES), "thresholds": params_yaml["risk_classifier"]["thresholds"],
                               "calibration": calibration, "provenance_hash": provenance_hash})
    save_model_artifact(artifact_dir / "risk_xgboost_default.joblib", model=default_model, features=features,
                        model_name="BhuRakshak risk XGBoost default baseline", model_version="v2.0.0-dev", **common,
                        extra={"class_names": list(CLASSES), "provenance_hash": provenance_hash})
    save_model_artifact(artifact_dir / "risk_logistic_baseline.joblib", model=logistic, features=features,
                        model_name="BhuRakshak risk logistic baseline", model_version="v2.0.0-dev", scaler=scaler,
                        **common, extra={"class_names": list(logistic.classes_), "provenance_hash": provenance_hash})
    for name, model in (("iforest_default", default_if), ("iforest_tuned", tuned_if)):
        threshold = thresholds["default" if name == "iforest_default" else "tuned"]
        save_model_artifact(artifact_dir / f"{name}.joblib", model=model, features=if_features,
                            model_name=f"BhuRakshak {name}", model_version="v2.0.0-dev", **common,
                            extra={"score_threshold": threshold, "threshold_percentile": alpha,
                                   "provenance_hash": provenance_hash})
    print(json.dumps({"development_rows": len(dev), "training_events": int(train.event_id.nunique()), "healthy_train_rows": len(healthy_train),
                      "healthy_validation_rows": len(healthy_val), "features": len(features),
                      "if_thresholds": thresholds, "calibration_method": selected_method,
                      "xgboost_tree_count": tuned_estimators, "provenance_hash": provenance_hash,
                      "artifacts": str(artifact_dir.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
