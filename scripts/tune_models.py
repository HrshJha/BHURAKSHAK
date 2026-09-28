#!/usr/bin/env python3
"""Development-only grouped baselines and Optuna tuning for risk and anomaly models.

This program reads only ``features_v2.parquet`` and the development split
assignment. It filters to train/validation before fitting and never accesses
the one-use held-out corpus.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STORE = ROOT / "data/features/features_v2.parquet"
SPLITS = ROOT / "data/features/split_assignment.csv"
OUT = ROOT / "reports/tuning"
PARAMS = ROOT / "configs/model_params.yaml"
CLASSES = ("NORMAL", "WARNING", "CRITICAL")
SEED = 42
FOLDS = 3

from src.anomaly.isolation_forest import healthy_baseline_mask
from src.config import anomaly_config, risk_model_config
from src.features.provenance import model_input_allowlist
from src.risk.xgboost_model import _resolve_features


def development_frame() -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(STORE)
    split_map = pd.read_csv(SPLITS, usecols=["event_id", "split"])
    frame = frame.merge(split_map, on="event_id", how="inner", validate="many_to_one")
    frame = frame[frame.split.isin(("train", "validation"))].copy()
    frame = frame.dropna(subset=["risk_label", "anomaly_label"])
    feature_cols = list(model_input_allowlist(_resolve_features(frame)))
    feature_cols = [c for c in feature_cols if frame[c].notna().any()]
    if not feature_cols:
        raise RuntimeError("provenance allow-list produced no risk-model inputs")
    return frame.reset_index(drop=True), feature_cols


def grouped_folds(frame: pd.DataFrame):
    events = frame.groupby("event_id", sort=True)["risk_label"].agg(lambda x: x.mode().iloc[0])
    ids = events.index.to_numpy()
    y = events.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    return [(ids[a], ids[b]) for a, b in splitter.split(ids, y, groups=ids)]


def _fold_frames(frame: pd.DataFrame, tr_events: np.ndarray, va_events: np.ndarray):
    return frame[frame.event_id.isin(tr_events)], frame[frame.event_id.isin(va_events)]


def _macro_pr_auc(y_true: np.ndarray, prob: np.ndarray, labels: list[str]) -> float:
    y = np.column_stack([y_true == label for label in labels]).astype(int)
    return float(np.mean([average_precision_score(y[:, i], prob[:, i]) for i in range(len(labels))]))


def _metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray, labels: list[str]) -> dict:
    return {
        "pr_auc_macro": _macro_pr_auc(y, prob, labels),
        "recall_critical": float(recall_score(y, pred, labels=["CRITICAL"], average=None, zero_division=0)[0]),
        "f1_macro": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "false_alarm_rate_normal": float(np.mean(pred[y == "NORMAL"] != "NORMAL")) if np.any(y == "NORMAL") else float("nan"),
    }


def _threshold_rule(val: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Run the existing configured rule and expose deterministic one-hot scores."""
    from src.risk.baselines import threshold_rule_predict
    pred = np.asarray(threshold_rule_predict(val), dtype=str)
    prob = np.zeros((len(pred), len(labels)), dtype=float)
    for j, label in enumerate(labels):
        prob[:, j] = (pred == label).astype(float)
    return pred, prob


def run_baselines(frame: pd.DataFrame, features: list[str]) -> dict:
    labels = list(CLASSES)
    records = {k: [] for k in ("threshold_rule", "logistic", "xgboost_default", "isolation_forest_default", "forecaster_persistence")}
    cfg = risk_model_config()
    if_cfg = anomaly_config()["isolation_forest"]
    for fold, (tr_ids, va_ids) in enumerate(grouped_folds(frame), 1):
        tr, va = _fold_frames(frame, tr_ids, va_ids)
        ytr, yv = tr.risk_label.astype(str).to_numpy(), va.risk_label.astype(str).to_numpy()
        rule_pred, rule_prob = _threshold_rule(va, labels)
        records["threshold_rule"].append(_metrics(yv, rule_pred, rule_prob, labels))

        scaler = StandardScaler().fit(tr[features].to_numpy(dtype=float))
        xtr = scaler.transform(tr[features].to_numpy(dtype=float))
        xv = scaler.transform(va[features].to_numpy(dtype=float))
        lr_cfg = cfg["baselines"]["logistic"]
        lr = LogisticRegression(C=float(lr_cfg["C"]), max_iter=int(lr_cfg["max_iter"]), class_weight="balanced", random_state=SEED)
        lr.fit(xtr, ytr)
        lrp = lr.predict_proba(xv)
        lrlabels = list(lr.classes_)
        aligned = np.zeros((len(va), len(labels)))
        for j, label in enumerate(lrlabels):
            aligned[:, labels.index(label)] = lrp[:, j]
        lrpred = np.asarray(labels)[aligned.argmax(axis=1)]
        records["logistic"].append(_metrics(yv, lrpred, aligned, labels))

        xgb_cfg = dict(cfg["xgboost"])
        xgb_cfg.update(objective="multi:softprob", num_class=3, eval_metric="mlogloss", tree_method="hist", n_jobs=1)
        xgb_cfg.pop("random_state", None)
        xgb = XGBClassifier(**xgb_cfg, random_state=SEED)
        ytr_i = np.asarray([labels.index(v) for v in ytr], dtype=int)
        xgb.fit(tr[features].to_numpy(dtype=float), ytr_i)
        xp = xgb.predict_proba(va[features].to_numpy(dtype=float))
        xprob = np.zeros((len(va), len(labels)))
        for j, label in enumerate(xgb.classes_):
            xprob[:, int(label)] = xp[:, j]
        xpred = np.asarray(labels)[xprob.argmax(axis=1)]
        records["xgboost_default"].append(_metrics(yv, xpred, xprob, labels))

        if_features = [c for c in features if frame[c].notna().any()]
        healthy_tr = tr[healthy_baseline_mask(tr)]
        if_model = IsolationForest(
            n_estimators=int(if_cfg["n_estimators"]), contamination="auto", random_state=SEED,
            n_jobs=1,
        ).fit(healthy_tr[if_features].to_numpy(dtype=float))
        score = -if_model.score_samples(va[if_features].to_numpy(dtype=float))
        y_anom = va.anomaly_label.to_numpy(dtype=int)
        healthy_val = va.loc[healthy_baseline_mask(va), if_features]
        threshold = float(np.quantile(-if_model.score_samples(healthy_val.to_numpy(dtype=float)), float(anomaly_config()["far_alpha"])))
        records["isolation_forest_default"].append({
            "pr_auc_anomaly": float(average_precision_score(y_anom, score)),
            "recall_at_validation_healthy_threshold": float(np.mean(score[y_anom == 1] > threshold)) if (y_anom == 1).any() else 0.0,
            "threshold": threshold,
            "healthy_train_rows": int(len(healthy_tr)),
        })

        # The free persistence baseline copies the last observed channel mean.
        # It is measured by the separate forecaster search where sequences are assembled.
        records["forecaster_persistence"].append({"status": "measured_in_forecaster_study"})

    return {
        "protocol": {"folds": FOLDS, "grouping": "event_id", "splitter": "StratifiedGroupKFold",
                     "seed": SEED, "development_rows": int(len(frame)), "development_events": int(frame.event_id.nunique()),
                     "test_touched": False, "features": features},
        "models": {name: {"folds": values, "mean": _mean_records(values)} for name, values in records.items()},
    }


def _mean_records(values: list[dict]) -> dict:
    keys = set.intersection(*(set(v) for v in values)) if values else set()
    result = {}
    for key in keys:
        nums = [v[key] for v in values if isinstance(v[key], (int, float)) and np.isfinite(v[key])]
        if nums:
            result[key] = {"mean": float(np.mean(nums)), "std": float(np.std(nums, ddof=1)) if len(nums) > 1 else 0.0}
    return result


def tune_xgboost(frame: pd.DataFrame, features: list[str], trial_limit: int, timeout: int) -> dict:
    import optuna
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler

    labels = list(CLASSES)
    folds = grouped_folds(frame)
    # Fixed fold-specific LR false-alarm floor is a tuning constraint.
    lr_far = []
    for tr_ids, va_ids in folds:
        tr, va = _fold_frames(frame, tr_ids, va_ids)
        ytr = tr.risk_label.astype(str).to_numpy()
        yv = va.risk_label.astype(str).to_numpy()
        scaler = StandardScaler().fit(tr[features].to_numpy(dtype=float))
        lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED)
        lr.fit(scaler.transform(tr[features].to_numpy(dtype=float)), ytr)
        pr = lr.predict(scaler.transform(va[features].to_numpy(dtype=float)))
        lr_far.append(float(np.mean(pr[yv == "NORMAL"] != "NORMAL")))

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
            "max_delta_step": trial.suggest_float("max_delta_step", 0, 10),
            "max_bin": trial.suggest_int("max_bin", 64, 512),
            "class_weighting": trial.suggest_categorical("class_weighting", ["none", "tempered", "inverse"]),
            "critical_threshold": trial.suggest_float("critical_threshold", 0.25, 0.75),
            "warning_threshold": trial.suggest_float("warning_threshold", 0.25, 0.75),
        }
        fold_scores = []
        far_values = []
        best_iterations = []
        for fi, (tr_ids, va_ids) in enumerate(folds):
            tr, va = _fold_frames(frame, tr_ids, va_ids)
            ytr = tr.risk_label.astype(str).to_numpy()
            yv = va.risk_label.astype(str).to_numpy()
            xtr = tr[features].to_numpy(dtype=float)
            xv = va[features].to_numpy(dtype=float)
            _, counts = np.unique(ytr, return_counts=True)
            per_class = {c: int(np.sum(ytr == c)) for c in labels}
            total = len(ytr)
            if params["class_weighting"] == "none":
                weights = np.ones(len(ytr), dtype=float)
            else:
                inverse = {c: total / (len(labels) * max(per_class[c], 1)) for c in labels}
                power = 1.0 if params["class_weighting"] == "inverse" else 0.5
                weights = np.array([inverse[c] ** power for c in ytr])
            model = XGBClassifier(
                **{k: v for k, v in params.items() if k not in {"class_weighting", "critical_threshold", "warning_threshold"}},
                n_estimators=2000, objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                tree_method="hist", early_stopping_rounds=50, n_jobs=1, random_state=SEED + fi,
            )
            ytr_i = np.asarray([labels.index(v) for v in ytr], dtype=int)
            yv_i = np.asarray([labels.index(v) for v in yv], dtype=int)
            model.fit(xtr, ytr_i, sample_weight=weights, eval_set=[(xv, yv_i)], verbose=False)
            best_iterations.append(int(getattr(model, "best_iteration", 0)) + 1)
            rawp = model.predict_proba(xv)
            aligned = np.zeros((len(va), len(labels)))
            for j, cl in enumerate(model.classes_):
                aligned[:, int(cl)] = rawp[:, j]
            pred = np.asarray(labels)[aligned.argmax(axis=1)]
            critical_i, warning_i = labels.index("CRITICAL"), labels.index("WARNING")
            pred[aligned[:, critical_i] >= params["critical_threshold"]] = "CRITICAL"
            mask_crit = pred != "CRITICAL"
            pred[mask_crit & (aligned[:, warning_i] >= params["warning_threshold"])] = "WARNING"
            m = _metrics(yv, pred, aligned, labels)
            far_values.append(m["false_alarm_rate_normal"])
            score = 0.5 * m["pr_auc_macro"] + 0.3 * m["recall_critical"] + 0.2 * m["f1_macro"]
            fold_scores.append(score)
            trial.report(float(np.mean(fold_scores)), step=fi)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_false_alarm_rates", far_values)
        trial.set_user_attr("best_iterations", best_iterations)
        trial.set_user_attr("false_alarm_constraint_pass", bool(np.all(np.array(far_values) <= np.array(lr_far) + 1e-12)))
        if not trial.user_attrs["false_alarm_constraint_pass"]:
            return -1.0
        trial.set_user_attr("lr_false_alarm_rates", lr_far)
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED), pruner=HyperbandPruner())
    start = time.monotonic()
    study.optimize(objective, n_trials=trial_limit, timeout=timeout, gc_after_trial=True, show_progress_bar=False)
    trials = []
    for t in study.trials:
        trials.append({"number": t.number, "state": t.state.name, "value": t.value, "params": t.params,
                       "user_attrs": t.user_attrs, "duration_seconds": t.duration.total_seconds() if t.duration else None})
    complete = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None and t.value >= 0]
    payload = {"study": {"sampler": "TPESampler", "pruner": "HyperbandPruner", "seed": SEED,
                          "requested_trials": trial_limit, "actual_trials": len(study.trials),
                          "completed_trials": len(complete), "timeout_seconds": timeout,
                          "elapsed_seconds": time.monotonic() - start, "folds": FOLDS,
                          "grouping": "event_id", "test_touched": False,
                          "objective": "0.5*macro_PR_AUC + 0.3*recall_CRITICAL + 0.2*macro_F1",
                          "constraint": "each fold FAR_NORMAL <= logistic regression FAR_NORMAL"},
               "best": None if not complete else {"value": study.best_value, "params": study.best_params,
                                                   "best_iterations": study.best_trial.user_attrs.get("best_iterations", [])},
               "trials": trials}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "xgboost_study.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload


def tune_iforest(frame: pd.DataFrame, features: list[str], trial_limit: int, timeout: int) -> dict:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    folds = grouped_folds(frame)
    # anomaly PR-AUC is the primary objective; tie-break is recall at 1% FPR.
    def objective(trial):
        params = {"n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                  "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
                  "max_features": trial.suggest_float("max_features", 0.5, 1.0),
                  "bootstrap": trial.suggest_categorical("bootstrap", [False, True])}
        aps, recalls = [], []
        for fi, (tr_ids, va_ids) in enumerate(folds):
            tr, va = _fold_frames(frame, tr_ids, va_ids)
            healthy = tr.loc[healthy_baseline_mask(tr), features]
            model = IsolationForest(**params, contamination="auto", random_state=SEED + fi, n_jobs=1)
            model.fit(healthy.to_numpy(dtype=float))
            score = -model.score_samples(va[features].to_numpy(dtype=float))
            target = va.anomaly_label.to_numpy(dtype=int)
            aps.append(float(average_precision_score(target, score)))
            normal = va.loc[healthy_baseline_mask(va), features]
            threshold = float(np.quantile(-model.score_samples(normal.to_numpy(dtype=float)), 0.99))
            recalls.append(float(np.mean(score[target == 1] > threshold)) if np.any(target == 1) else 0.0)
            trial.report(float(np.mean(aps)), step=fi)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_recall_at_1pct_fpr", recalls)
        trial.set_user_attr("mean_recall_at_1pct_fpr_tiebreak", float(np.mean(recalls)))
        return float(np.mean(aps))

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED), pruner=MedianPruner(n_startup_trials=10))
    start = time.monotonic()
    study.optimize(objective, n_trials=trial_limit, timeout=timeout, gc_after_trial=True, show_progress_bar=False)
    trials = [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params,
               "user_attrs": t.user_attrs, "duration_seconds": t.duration.total_seconds() if t.duration else None}
              for t in study.trials]
    complete = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
    best = None
    if complete:
        best_trial = max(complete, key=lambda t: (t.value, t.user_attrs.get("mean_recall_at_1pct_fpr_tiebreak", 0)))
        best = {"value": best_trial.value, "params": best_trial.params, "user_attrs": best_trial.user_attrs}
    payload = {"study": {"sampler": "TPESampler", "pruner": "MedianPruner", "seed": SEED,
                          "requested_trials": trial_limit, "actual_trials": len(study.trials),
                          "completed_trials": len(complete), "timeout_seconds": timeout,
                          "elapsed_seconds": time.monotonic() - start, "folds": FOLDS,
                          "grouping": "event_id", "healthy_fit_only": True, "test_touched": False},
               "best": best, "trials": trials}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "isolation_forest_study.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baselines", "xgboost", "iforest"), required=True)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    frame, features = development_frame()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "baselines":
        result = run_baselines(frame, features)
        (OUT / "baselines.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"events": result["protocol"]["development_events"], "rows": result["protocol"]["development_rows"], "features": len(features)}))
        return 0
    count = args.trials or (200 if args.phase == "xgboost" else 100)
    if args.phase == "xgboost":
        result = tune_xgboost(frame, features, count, args.timeout_seconds)
    else:
        result = tune_iforest(frame, features, count, args.timeout_seconds)
    print(json.dumps(result["study"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
