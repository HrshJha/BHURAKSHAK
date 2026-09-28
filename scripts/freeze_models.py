#!/usr/bin/env python3
"""Fit and register the selected development-only model bundles for Phase 8."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tune_models import CLASSES, development_frame
from src.anomaly.isolation_forest import healthy_baseline_mask
from src.config import anomaly_config, risk_model_config
from src.risk.artifacts import save_model_artifact


def _weight_vector(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(len(y), dtype=float)
    counts = {c: max(1, int(np.sum(y == c))) for c in CLASSES}
    base = {c: len(y) / (len(CLASSES) * counts[c]) for c in CLASSES}
    power = 1.0 if mode == "inverse" else 0.5
    return np.array([base[c] ** power for c in y], dtype=float)


def _fit_xgb(df: pd.DataFrame, features: list[str], params: dict, version: str, sample_weight: np.ndarray | None = None):
    y = df.risk_label.astype(str).to_numpy()
    numeric = np.asarray([CLASSES.index(v) for v in y], dtype=int)
    clf = XGBClassifier(
        **params, n_estimators=2000, objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", tree_method="hist", n_jobs=1, random_state=42,
    )
    clf.fit(df[features].to_numpy(dtype=float), numeric, sample_weight=sample_weight)
    return clf


def main() -> int:
    baseline_report = json.loads((ROOT / "reports/tuning/baselines.json").read_text(encoding="utf-8"))
    xgb_study = json.loads((ROOT / "reports/tuning/xgboost_study.json").read_text(encoding="utf-8"))
    if_study = json.loads((ROOT / "reports/tuning/isolation_forest_study.json").read_text(encoding="utf-8"))
    forecaster_path = ROOT / "reports/tuning/forecaster_study.json"
    if not forecaster_path.exists():
        raise SystemExit("forecaster Optuna study is required before freezing models")
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
    risk_model = _fit_xgb(dev, features, xgb_params, "tuned", _weight_vector(dev.risk_label.astype(str).to_numpy(), weight_mode))

    base_cfg = dict(risk_model_config()["xgboost"])
    base_cfg.pop("random_state", None)
    base_cfg = {k: v for k, v in base_cfg.items() if k != "n_estimators"}
    default_model = _fit_xgb(dev, features, base_cfg, "default")

    scaler = StandardScaler().fit(dev[features].to_numpy(dtype=float))
    logistic_cfg = risk_model_config()["baselines"]["logistic"]
    logistic = LogisticRegression(C=float(logistic_cfg["C"]), max_iter=int(logistic_cfg["max_iter"]),
                                  class_weight="balanced", random_state=42)
    logistic.fit(scaler.transform(dev[features].to_numpy(dtype=float)), dev.risk_label.astype(str).to_numpy())

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
        "risk_classifier": {
            "status": "frozen_after_development_grouped_cv",
            "params": best_xgb["params"], "features": features,
            "thresholds": {"critical": best_xgb["params"]["critical_threshold"],
                           "warning": best_xgb["params"]["warning_threshold"]},
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
        "decision_rule": risk_model_config()["baselines"]["threshold_rule"],
        "training": {"dataset_version": dataset_version, "feature_schema_version": "v2",
                     "seed": 42, "groups": int(dev.event_id.nunique()),
                     "split": "train+validation after grouped development CV", "test_touched": False},
    }
    (ROOT / "configs/model_params.yaml").write_text(yaml.safe_dump(params_yaml, sort_keys=False), encoding="utf-8")
    artifact_dir = ROOT / "models" / "tuned"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    common = {"training_dataset_version": dataset_version, "split_name": "train+validation", "seed": 42}
    save_model_artifact(artifact_dir / "risk_xgboost_tuned.joblib", model=risk_model, features=features,
                        model_name="SubSense risk XGBoost tuned", model_version="v2.0.0-dev", **common,
                        extra={"class_names": list(CLASSES), "thresholds": params_yaml["risk_classifier"]["thresholds"]})
    save_model_artifact(artifact_dir / "risk_xgboost_default.joblib", model=default_model, features=features,
                        model_name="SubSense risk XGBoost default baseline", model_version="v2.0.0-dev", **common,
                        extra={"class_names": list(CLASSES)})
    save_model_artifact(artifact_dir / "risk_logistic_baseline.joblib", model=logistic, features=features,
                        model_name="SubSense risk logistic baseline", model_version="v2.0.0-dev", scaler=scaler,
                        **common, extra={"class_names": list(logistic.classes_)})
    for name, model in (("iforest_default", default_if), ("iforest_tuned", tuned_if)):
        threshold = thresholds["default" if name == "iforest_default" else "tuned"]
        save_model_artifact(artifact_dir / f"{name}.joblib", model=model, features=if_features,
                            model_name=f"SubSense {name}", model_version="v2.0.0-dev", **common,
                            extra={"score_threshold": threshold, "threshold_percentile": alpha})
    print(json.dumps({"development_rows": len(dev), "healthy_train_rows": len(healthy_train),
                      "healthy_validation_rows": len(healthy_val), "features": len(features),
                      "if_thresholds": thresholds, "artifacts": str(artifact_dir.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
