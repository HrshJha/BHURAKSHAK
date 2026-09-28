#!/usr/bin/env python3
"""Re-evaluate selected tuned candidates on the fixed development CV folds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, recall_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tune_models import CLASSES, _metrics, development_frame, grouped_folds
from src.anomaly.isolation_forest import healthy_baseline_mask


def _weights(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(len(y), dtype=float)
    counts = {c: max(1, int(np.sum(y == c))) for c in CLASSES}
    power = 1.0 if mode == "inverse" else 0.5
    return np.array([(len(y) / (len(CLASSES) * counts[c])) ** power for c in y])


def _pred(prob: np.ndarray, critical: float, warning: float) -> np.ndarray:
    result = np.asarray(CLASSES)[prob.argmax(axis=1)].copy()
    result[prob[:, 2] >= critical] = "CRITICAL"
    result[(result != "CRITICAL") & (prob[:, 1] >= warning)] = "WARNING"
    return result


def _mean_std(records: list[dict], keys: tuple[str, ...]) -> dict:
    return {key: {"mean": float(np.mean([r[key] for r in records])),
                  "std": float(np.std([r[key] for r in records], ddof=1)) if len(records) > 1 else 0.0}
            for key in keys}


def main() -> int:
    frame, features = development_frame()
    xgb_study = json.loads((ROOT / "reports/tuning/xgboost_study.json").read_text(encoding="utf-8"))
    if_study = json.loads((ROOT / "reports/tuning/isolation_forest_study.json").read_text(encoding="utf-8"))
    xbest, ibest = xgb_study["best"], if_study["best"]
    params = dict(xbest["params"])
    weight_mode = params.pop("class_weighting")
    critical, warning = float(params.pop("critical_threshold")), float(params.pop("warning_threshold"))
    xgb_folds, if_folds = [], []
    for fi, (train_ids, val_ids) in enumerate(grouped_folds(frame)):
        train = frame[frame.event_id.isin(train_ids)]
        val = frame[frame.event_id.isin(val_ids)]
        y_train = train.risk_label.astype(str).to_numpy()
        y_val = val.risk_label.astype(str).to_numpy()
        n_estimators = int(xbest.get("best_iterations", [])[fi]) if xbest.get("best_iterations") else 500
        model = XGBClassifier(**params, n_estimators=n_estimators, objective="multi:softprob", num_class=3,
                              eval_metric="mlogloss", tree_method="hist", n_jobs=1, random_state=42 + fi)
        y_numeric = np.asarray([CLASSES.index(c) for c in y_train], dtype=int)
        model.fit(train[features].to_numpy(dtype=float), y_numeric,
                  sample_weight=_weights(y_train, weight_mode))
        raw = model.predict_proba(val[features].to_numpy(dtype=float))
        prob = np.zeros((len(val), len(CLASSES)))
        for j, class_id in enumerate(model.classes_):
            prob[:, int(class_id)] = raw[:, j]
        xgb_folds.append({"fold": fi + 1, "n_estimators": n_estimators,
                          **_metrics(y_val, _pred(prob, critical, warning), prob, list(CLASSES))})

        if_params = dict(ibest["params"])
        healthy_train = train.loc[healthy_baseline_mask(train), features]
        healthy_val = val.loc[healthy_baseline_mask(val), features]
        anomaly = IsolationForest(**if_params, contamination="auto", random_state=42 + fi, n_jobs=1)
        anomaly.fit(healthy_train.to_numpy(dtype=float))
        scores = -anomaly.score_samples(val[features].to_numpy(dtype=float))
        y_anomaly = val.anomaly_label.to_numpy(dtype=int)
        threshold = float(np.quantile(-anomaly.score_samples(healthy_val.to_numpy(dtype=float)), 0.99))
        if_folds.append({"fold": fi + 1, "pr_auc_anomaly": float(average_precision_score(y_anomaly, scores)),
                         "recall_at_validation_healthy_99th_percentile": float(np.mean(scores[y_anomaly == 1] > threshold)) if (y_anomaly == 1).any() else 0.0,
                         "false_positive_rate_healthy": float(np.mean(scores[y_anomaly == 0] > threshold)) if (y_anomaly == 0).any() else 0.0,
                         "threshold": threshold})

    baseline = json.loads((ROOT / "reports/tuning/baselines.json").read_text(encoding="utf-8"))
    result = {
        "protocol": {"splitter": "StratifiedGroupKFold", "folds": 3, "group": "event_id",
                     "development_only": True, "test_touched": False, "feature_count": len(features),
                     "window_count": int(len(frame)), "event_count": int(frame.event_id.nunique())},
        "study_counts": {"xgboost": xgb_study["study"], "isolation_forest": if_study["study"]},
        "risk_models": {"threshold_rule": baseline["models"]["threshold_rule"],
                        "logistic_regression": baseline["models"]["logistic"],
                        "xgboost_default": baseline["models"]["xgboost_default"],
                        "xgboost_tuned": {"folds": xgb_folds,
                                          "mean": _mean_std(xgb_folds, ("pr_auc_macro", "recall_critical", "f1_macro", "false_alarm_rate_normal")),
                                          "objective": xbest["value"], "params": xbest["params"]}},
        "anomaly_models": {"isolation_forest_default": baseline["models"]["isolation_forest_default"],
                           "isolation_forest_tuned": {"folds": if_folds,
                                                       "mean": _mean_std(if_folds, ("pr_auc_anomaly", "recall_at_validation_healthy_99th_percentile", "false_positive_rate_healthy")),
                                                       "params": ibest["params"]}},
    }
    lr_folds = baseline["models"]["logistic"]["folds"]
    result["xgboost_false_alarm_constraint_revalidation"] = {
        "comparator": "selected scaled balanced logistic baseline",
        "logistic_params": baseline["models"]["logistic"].get("params", {}),
        "folds": [{"fold": int(xgb_folds[i]["fold"]),
                   "xgboost_false_alarm_rate_normal": float(xgb_folds[i]["false_alarm_rate_normal"]),
                   "logistic_false_alarm_rate_normal": float(lr_folds[i]["false_alarm_rate_normal"]),
                   "passes": bool(xgb_folds[i]["false_alarm_rate_normal"] <= lr_folds[i]["false_alarm_rate_normal"] + 1e-12)}
                  for i in range(min(len(xgb_folds), len(lr_folds)))],
    }
    forecaster_path = ROOT / "reports/tuning/forecaster_study.json"
    if forecaster_path.exists():
        forecast = json.loads(forecaster_path.read_text(encoding="utf-8"))
        result["forecaster"] = {"status": "tuned" if forecast.get("beats_persistence") else "did_not_beat_persistence",
                                 "study": forecast["study"], "best": forecast["best"],
                                 "top3_repeated_cv": forecast.get("top3_repeated_cv", [])}
    out = ROOT / "reports/tuning/tuned_cv_metrics.json"
    out.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}; tuned XGBoost objective={xbest['value']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
