#!/usr/bin/env python3
"""Required model sanity, attribution, ablation, and workstation profiling."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tune_models import CLASSES, _metrics, development_frame, grouped_folds
from src.risk.artifacts import load_model_artifact


def _weight_vector(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(len(y))
    counts = {c: max(1, int(np.sum(y == c))) for c in CLASSES}
    power = 1.0 if mode == "inverse" else 0.5
    return np.array([(len(y) / (len(CLASSES) * counts[c])) ** power for c in y])


def _fit(frame: pd.DataFrame, features: list[str], params: dict, n_estimators: int, seed: int):
    from xgboost import XGBClassifier
    params = dict(params)
    mode = str(params.pop("class_weighting", "none"))
    params.pop("critical_threshold", None)
    params.pop("warning_threshold", None)
    y = frame.risk_label.astype(str).to_numpy()
    y_num = np.asarray([CLASSES.index(c) for c in y], dtype=int)
    model = XGBClassifier(**params, n_estimators=n_estimators, objective="multi:softprob", num_class=3,
                          eval_metric="mlogloss", tree_method="hist", n_jobs=1, random_state=seed)
    model.fit(frame[features].to_numpy(dtype=float), y_num, sample_weight=_weight_vector(y, mode))
    return model


def _prob(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    raw = model.predict_proba(frame[features].to_numpy(dtype=float))
    out = np.zeros((len(frame), len(CLASSES)))
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, j]
    return out


def _pred(prob: np.ndarray, thresholds: dict | None = None) -> np.ndarray:
    pred = np.asarray(CLASSES)[prob.argmax(axis=1)].copy()
    if thresholds:
        ci, wi = CLASSES.index("CRITICAL"), CLASSES.index("WARNING")
        pred[prob[:, ci] >= float(thresholds["critical"])] = "CRITICAL"
        pred[(pred != "CRITICAL") & (prob[:, wi] >= float(thresholds["warning"]))] = "WARNING"
    return pred


def _macro_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([average_precision_score((y == c).astype(int), p[:, i]) for i, c in enumerate(CLASSES)]))


def _latency_profile(fn, repeats: int = 1000) -> dict:
    proc = psutil.Process()
    fn()  # warm up native dispatch before recording steady-state latency
    rss_before = int(proc.memory_info().rss)
    peak = rss_before
    times = []
    for i in range(repeats):
        start = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - start) / 1e6)
        if i % 20 == 0:
            peak = max(peak, int(proc.memory_info().rss))
    rss_after = int(proc.memory_info().rss)
    peak = max(peak, rss_after)
    return {"repeats": repeats, "latency_ms_p50": float(np.quantile(times, 0.5)),
            "latency_ms_p95": float(np.quantile(times, 0.95)),
            "process_rss_before_bytes": rss_before, "process_rss_after_bytes": rss_after,
            "process_rss_peak_sampled_bytes": peak}


def main() -> int:
    frame, features = development_frame()
    train = frame[frame.split == "train"].copy().reset_index(drop=True)
    val = frame[frame.split == "validation"].copy().reset_index(drop=True)
    config = yaml.safe_load((ROOT / "configs/model_params.yaml").read_text(encoding="utf-8"))
    classifier = config["risk_classifier"]
    artifact = load_model_artifact(ROOT / "models/tuned/risk_xgboost_tuned.joblib", expected_schema_version="v2")
    model = artifact["model"]
    feature_set = list(artifact["features"])
    thresholds = classifier["thresholds"]
    p_val = _prob(model, val, feature_set)
    y_val = val.risk_label.astype(str).to_numpy()
    result: dict = {"protocol": {"train_events": int(train.event_id.nunique()),
                                  "validation_events": int(val.event_id.nunique()),
                                  "features": feature_set, "test_touched": False}}
    output_path = ROOT / "reports/tuning/robustness.json"

    def persist_result() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")

    # Label permutation control: shuffle train targets after the grouped split.
    shuffle_records = []
    for fold, (tr_ids, va_ids) in enumerate(grouped_folds(frame), 1):
        tr = frame[frame.event_id.isin(tr_ids)].copy()
        va = frame[frame.event_id.isin(va_ids)]
        rng = np.random.default_rng(42 + fold)
        y_original = tr.risk_label.astype(str).to_numpy()
        shuffled = rng.permutation(y_original)
        tr["risk_label"] = shuffled
        fitted = _fit(tr, feature_set, classifier["params"], int(classifier["n_estimators"]), 42 + fold)
        p = _prob(fitted, va, feature_set)
        pred = _pred(p, thresholds)
        m = _metrics(va.risk_label.astype(str).to_numpy(), pred, p, list(CLASSES))
        shuffle_records.append(m)
    result["shuffled_label_control"] = {"folds": shuffle_records,
                                           "mean_macro_pr_auc": float(np.mean([m["pr_auc_macro"] for m in shuffle_records])),
                                           "mean_macro_f1": float(np.mean([m["f1_macro"] for m in shuffle_records])),
                                           "chance_reference": "macro PR-AUC near 1/3; interpretation depends on class prevalence"}
    persist_result()

    # Learning curve on event-disjoint train/validation partitions. Fractions
    # are sampled within event-level risk strata, never at the window level.
    train_event_y = train.groupby("event_id", sort=True).risk_label.agg(lambda s: s.mode().iloc[0])
    rng = np.random.default_rng(42)
    learning_curve = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        selected = []
        for label in CLASSES:
            ids = train_event_y[train_event_y == label].index.to_numpy().copy()
            rng.shuffle(ids)
            selected.extend(ids[:max(1, int(round(len(ids) * fraction)))])
        subset = train[train.event_id.isin(selected)]
        fitted = _fit(subset, feature_set, classifier["params"], int(classifier["n_estimators"]), 42)
        p_train = _prob(fitted, subset, feature_set)
        p_val_curve = _prob(fitted, val, feature_set)
        m_train = _metrics(subset.risk_label.astype(str).to_numpy(), _pred(p_train), p_train, list(CLASSES))
        m_val = _metrics(y_val, _pred(p_val_curve), p_val_curve, list(CLASSES))
        learning_curve.append({"fraction_of_training_events": fraction,
                               "training_events": int(subset.event_id.nunique()),
                               "train_metrics": m_train, "validation_metrics": m_val,
                               "macro_pr_auc_gap_train_minus_validation": m_train["pr_auc_macro"] - m_val["pr_auc_macro"]})
    result["learning_curve"] = learning_curve
    persist_result()

    # Full feature-group ablations using only active, provenance-approved inputs.
    groups = yaml.safe_load((ROOT / "configs/feature_schema_v2.yaml").read_text(encoding="utf-8"))["feature_groups"]
    group_results = {}
    for group, declared in groups.items():
        dropped = [c for c in feature_set if c in set(declared)]
        if not dropped:
            group_results[group] = {"status": "no_active_features (gated or unavailable)", "dropped": []}
            continue
        kept = [c for c in feature_set if c not in dropped]
        fitted = _fit(train, kept, classifier["params"], int(classifier["n_estimators"]), 42)
        prob = _prob(fitted, val, kept)
        pred = _pred(prob, thresholds)
        group_results[group] = {"dropped": dropped, "metrics": _metrics(y_val, pred, prob, list(CLASSES))}
    result["feature_group_ablations"] = group_results
    persist_result()

    # Isolation Forest input-group ablation: fit only healthy training rows and
    # choose each comparison threshold from healthy validation rows.
    if_params = dict(config["isolation_forest"]["params"])
    healthy_train = train.loc[train.anomaly_label.eq(0)]
    healthy_val = val.loc[val.anomaly_label.eq(0)]
    if_ablation = {}
    for group, declared in groups.items():
        dropped = [c for c in feature_set if c in set(declared)]
        kept = [c for c in feature_set if c not in dropped]
        if not dropped or not kept:
            if_ablation[group] = {"status": "no_active_features_to_drop", "dropped": dropped}
            continue
        fitted = __import__("sklearn.ensemble", fromlist=["IsolationForest"]).IsolationForest(
            **if_params, contamination="auto", random_state=42, n_jobs=1
        ).fit(healthy_train[kept].to_numpy(dtype=float))
        val_score = -fitted.score_samples(val[kept].to_numpy(dtype=float))
        val_healthy_score = -fitted.score_samples(healthy_val[kept].to_numpy(dtype=float))
        threshold = float(np.quantile(val_healthy_score, 0.99))
        target = val.anomaly_label.to_numpy(dtype=int)
        if_ablation[group] = {
            "dropped": dropped,
            "features_remaining": len(kept),
            "pr_auc_anomaly": float(average_precision_score(target, val_score)),
            "recall_at_healthy_validation_99th_percentile": float(np.mean(val_score[target == 1] > threshold)) if (target == 1).any() else 0.0,
            "threshold": threshold,
        }
    result["isolation_forest_feature_group_ablations"] = if_ablation
    persist_result()

    # Native XGBoost TreeSHAP contributions and permutation importance.
    rng = np.random.default_rng(42)
    sample_idx = np.sort(rng.choice(len(val), size=min(2000, len(val)), replace=False))
    sample = val.iloc[sample_idx]
    booster = model.get_booster()
    shap_values = booster.predict(__import__("xgboost").DMatrix(sample[feature_set].to_numpy(dtype=float)),
                                  pred_contribs=True, strict_shape=True)
    # XGBoost strict-shape multiclass: (rows, groups, features + bias).
    abs_values = np.abs(shap_values[..., :-1]).mean(axis=(0, 1))
    top_idx = np.argsort(abs_values)[::-1][:3]
    top_features = [{"feature": feature_set[i], "mean_abs_shap": float(abs_values[i])} for i in top_idx]
    rng = np.random.default_rng(42)
    x_val = val[feature_set].to_numpy(dtype=float)
    base_ap = _macro_ap(y_val, p_val)
    permutation = []
    for j, name in enumerate(feature_set):
        scores = []
        for _ in range(3):
            shuffled_x = x_val.copy()
            shuffled_x[:, j] = rng.permutation(shuffled_x[:, j])
            p = _prob(model, pd.DataFrame(shuffled_x, columns=feature_set), feature_set)
            scores.append(base_ap - _macro_ap(y_val, p))
        permutation.append({"feature": name, "macro_pr_auc_drop_mean": float(np.mean(scores)),
                            "macro_pr_auc_drop_std": float(np.std(scores, ddof=1))})
    permutation.sort(key=lambda x: x["macro_pr_auc_drop_mean"], reverse=True)
    result["tree_shap"] = {"method": "XGBoost pred_contribs (TreeSHAP)", "sample_rows": len(sample),
                           "top3": top_features}
    result["permutation_importance"] = permutation
    persist_result()

    # Drop the top SHAP features one at a time and refit.
    top_ablation = []
    for item in top_features:
        dropped_feature = item["feature"]
        kept = [c for c in feature_set if c != dropped_feature]
        fitted = _fit(train, kept, classifier["params"], int(classifier["n_estimators"]), 42)
        p = _prob(fitted, val, kept)
        top_ablation.append({"dropped": dropped_feature, "metrics": _metrics(y_val, _pred(p, thresholds), p, list(CLASSES))})
    result["top_shap_feature_ablations"] = top_ablation
    persist_result()

    # Repeated grouped CV for the three highest-scoring feasible trial configs.
    study = json.loads((ROOT / "reports/tuning/xgboost_study.json").read_text(encoding="utf-8"))
    ranked = sorted([t for t in study["trials"] if t["state"] == "COMPLETE" and t["value"] is not None and t["value"] >= 0],
                    key=lambda t: float(t["value"]), reverse=True)
    unique, seen = [], set()
    for trial in ranked:
        key = json.dumps(trial["params"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(trial)
        if len(unique) == 3:
            break
    repeats = []
    for trial in unique:
        params = dict(trial["params"])
        for cv_seed in (42, 43, 44):
            event_y = frame.groupby("event_id", sort=True).risk_label.agg(lambda s: s.mode().iloc[0])
            ids = event_y.index.to_numpy()
            splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=cv_seed)
            fold_results = []
            for tr_idx, va_idx in splitter.split(ids, event_y.to_numpy(), groups=ids):
                tr = frame[frame.event_id.isin(ids[tr_idx])]
                va = frame[frame.event_id.isin(ids[va_idx])]
                fitted = _fit(tr, feature_set, params, int(classifier["n_estimators"]), cv_seed)
                p = _prob(fitted, va, feature_set)
                fold_results.append(_metrics(va.risk_label.astype(str).to_numpy(), _pred(p, thresholds), p, list(CLASSES)))
            repeats.append({"trial_number": trial["number"], "seed": cv_seed, "params": params, "folds": fold_results,
                            "mean": {k: float(np.mean([f[k] for f in fold_results])
                                             ) for k in ("pr_auc_macro", "recall_critical", "f1_macro", "false_alarm_rate_normal")}})
    result["top3_repeated_grouped_cv"] = repeats
    persist_result()

    # Scenario recall on validation events. Family IDs are generator metadata, not features.
    pred_val = _pred(p_val, thresholds)
    scenario = val.event_id.str.rsplit("_", n=2).str[0]
    per_scenario = {}
    for name, idx in val.groupby(scenario, sort=True).groups.items():
        rows = np.asarray(list(idx), dtype=int)
        yt, yp = y_val[rows], pred_val[rows]
        per_scenario[str(name)] = {"events": int(val.loc[rows, "event_id"].nunique()),
                                   "recall_per_class": {c: float(recall_score(yt, yp, labels=[c], average=None, zero_division=0)[0]) for c in CLASSES}}
    result["per_scenario_validation_recall"] = per_scenario
    scenario_alias = scenario.astype(str).str.replace(r"^E_", "", regex=True)
    scenario_sets = {
        "SENSOR_FAULT": {"sensor_bias", "sensor_stuck", "sensor_dropout", "sensor_spike", "sensor_drift"},
        "DATA_QUALITY": {"packet_loss", "communication_failure"},
        "COMMUNICATION_FAILURE": {"communication_failure"},
        "noise_injected": {"sensor_bias", "sensor_stuck", "sensor_dropout", "sensor_spike", "sensor_drift",
                           "packet_loss", "single_node_disturbance", "vibration_only", "slow_drift_temperature"},
    }
    category_recall = {}
    for category, members in scenario_sets.items():
        mask = scenario_alias.isin(members).to_numpy()
        yt, yp = y_val[mask], pred_val[mask]
        category_recall[category] = {"events": int(val.loc[mask, "event_id"].nunique()),
                                     "recall_per_class": {c: float(recall_score(yt, yp, labels=[c], average=None, zero_division=0)[0]) for c in CLASSES}}
    category_recall["missing_modality"] = {
        "status": "unavailable_in_this_synthetic_corpus",
        "available_source_versions": json.loads((ROOT / "data/synthetic/dataset_manifest.json").read_text()).get("source_data_versions", {}),
        "note": "Sentinel-1 and DGPS source channels are not present in this development corpus.",
    }
    result["required_scenario_category_recall"] = category_recall
    persist_result()

    # Workstation-only inference latency and process RSS on a single window.
    probe = val[feature_set].iloc[[0]].to_numpy(dtype=float)
    profile_models = {"risk_xgboost_tuned": _latency_profile(lambda: model.predict_proba(probe))}
    if_bundle = load_model_artifact(ROOT / "models/tuned/iforest_tuned.joblib", expected_schema_version="v2")
    if_probe = if_bundle["transform"](val).iloc[[0]].to_numpy(dtype=float)
    profile_models["isolation_forest_tuned"] = _latency_profile(lambda: if_bundle["model"].score_samples(if_probe))

    # Load, do not retrain, the frozen physical forecaster for an apples-to-
    # apples one-history-step workstation profile.
    forecast_path = ROOT / "models/temporal_model/forecaster_tuned.pt"
    if forecast_path.exists():
        import torch
        from torch import nn
        from src.forecasting.temporal_model import _build_torch
        checkpoint = torch.load(forecast_path, map_location="cpu", weights_only=False)
        fp = checkpoint["params"]
        channels = tuple(checkpoint["channels"])
        horizons = tuple(checkpoint["horizons"])
        width, depth, architecture = int(fp["width"]), int(fp["depth"]), str(fp["architecture"])
        class FrozenForecaster(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = _build_torch(architecture, len(channels), width, depth)
                self.head = nn.Linear(width, len(horizons) * len(channels))
                self.dropout = nn.Dropout(float(fp.get("dropout", 0.0)))
                if architecture == "tcn":
                    self.tcn_proj = nn.Linear(width, width)
            def forward(self, x):
                if architecture == "tcn":
                    last = self.backbone(x.transpose(1, 2))[:, :, -1]
                else:
                    out, _ = self.backbone(x)
                    last = out[:, -1, :]
                return self.head(self.dropout(last)).reshape(-1, len(horizons), len(channels))
        forecaster = FrozenForecaster()
        forecaster.load_state_dict(checkpoint["state_dict"])
        forecaster.eval()
        fframe = pd.read_parquet(ROOT / "data/features/features_v2.parquet")
        first_event = fframe.event_id.iloc[0]
        series = fframe[fframe.event_id == first_event].sort_values("window_index")
        # The persisted feature store already carries Group A's window means
        # under their canonical names (displacement, tilt_x, tilt_y).
        raw_sequence = series[list(channels)].tail(int(fp["history_steps"])).to_numpy(dtype=np.float32)
        x = torch.tensor(((raw_sequence - checkpoint["mu"]) / checkpoint["sd"])[None, ...], dtype=torch.float32)
        with torch.no_grad():
            profile_models["forecaster_tuned"] = _latency_profile(lambda: forecaster(x))
    else:
        profile_models["forecaster_tuned"] = {"status": "no_frozen_forecaster_artifact"}
    result["workstation_edge_profile"] = {
        "platform": sys.platform,
        "models": profile_models,
        "note": "workstation CPU/process RSS; sampled RSS is not device peak; no numeric edge budget is defined in  ",
    }
    persist_result()
    print(f"wrote {output_path.relative_to(ROOT)}; shuffled macro PR-AUC={result['shuffled_label_control']['mean_macro_pr_auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
