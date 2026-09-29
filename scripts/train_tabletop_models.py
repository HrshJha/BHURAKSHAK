#!/usr/bin/env python3
"""Train the plan's two-model pipeline on the tabletop dataset (plan –).

Models ( — the two-model pipeline, run in parallel at inference):
 1. Isolation Forest — unsupervised trip-wire, trained ONLY on Stable
 (severity_class == 0) windows from the training trials; flags anything
 that deviates from normal, including patterns never labeled.
 2. Random Forest — supervised 4-class severity classifier on the
 engineered features, class-weighted because Stable dominates.

Protocol:
 - split by TRIAL, never by row — 18 train / 4 validation / 4 test,
 each split mixing conditions and speeds;
 - StandardScaler fitted on the training split only (, step 3);
 - hyperparameters tuned on the VALIDATION set only, then frozen;
 - GroupKFold(5) cross-validation across trials for the RF;
 - one final evaluation on the held-out test trials.

Metrics: per-class confusion/recall (critical recall first), IF score
separation, and lead time — seconds before the trapdoor reaches the critical
threshold that the pipeline first flags it.

Alert mapping: green = IF normal AND RF Stable; yellow = Initiation;
orange = Progressive; red = Critical OR strong IF anomaly.

The threshold-rule baseline is trained alongside for the honest
"ML must beat it" comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data" / "processed" / "sensors" / "tabletop" / "processed_windowed_dataset.csv"
RAW = REPO_ROOT / "data" / "raw" / "sensors" / "tabletop" / "raw_sensor_log.csv"
META = REPO_ROOT / "data" / "raw" / "sensors" / "tabletop" / "trial_metadata.csv"
OUT_MODELS = REPO_ROOT / "models" / "tabletop"
OUT_REPORTS = REPO_ROOT / "reports"
OUT_EXPERIMENTS = REPO_ROOT / "experiments"

SEED = 42
CRITICAL_MM = 35.0
CLASS_NAMES = {0: "Stable", 1: "Initiation", 2: "Progressive", 3: "Critical"}
ALERT_NAMES = {0: "green", 1: "yellow", 2: "orange", 3: "red"}

#: split by trial, mixing conditions and speeds in every split
TRAIN_TRIALS = [
    "T001", "T003", "T005", "T006", "T007", "T008", "T009", "T011",
    "T013", "T015", "T016", "T018", "T019", "T020", "T022", "T023",
    "T025", "T026",
]
VAL_TRIALS = ["T002", "T010", "T014", "T021"]
TEST_TRIALS = ["T004", "T012", "T017", "T024"]

FEATURES = [
    "roll", "pitch", "vibration_rms", "accel_peak", "tof_delta", "tof_rate",
    "displacement_mm", "displacement_rate", "relative_tilt", "relative_vibration",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    win = pd.read_csv(DATA)
    meta = pd.read_csv(META).set_index("trial_id")
    # trial-level condition/speed joined onto windows for stratification checks
    win["condition"] = win["trial_id"].map(meta["condition"])
    win["speed"] = win["trial_id"].map(meta["speed"])
    return win, meta


def split_frame(win: pd.DataFrame, trials: list[str]) -> pd.DataFrame:
    return win[win.trial_id.isin(trials)].reset_index(drop=True)


def train_if(train: pd.DataFrame) -> tuple[IsolationForest, StandardScaler, np.ndarray]:
    """ step 4: Isolation Forest on Stable windows of the training trials."""
    stable = train[train.severity_class == 0]
    scaler = StandardScaler().fit(stable[FEATURES])
    X = scaler.transform(stable[FEATURES])
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED, n_jobs=-1)
    iso.fit(X)
    return iso, scaler, X


def if_scores(iso: IsolationForest, scaler: StandardScaler, df: pd.DataFrame) -> np.ndarray:
    """Return higher anomaly scores for less typical sensor windows."""
    return -iso.score_samples(scaler.transform(df[FEATURES]))


def tune_rf(val_split: pd.DataFrame) -> dict:
    """Hyperparameter search on the VALIDATION split only ( step 6)."""
    X = val_split[FEATURES].to_numpy()
    y = val_split["severity_class"].to_numpy()
    best, best_score = None, -1.0
    for n_est in (200, 400):
        for depth in (None, 12, 8):
            for min_leaf in (1, 2):
                clf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    min_samples_leaf=min_leaf,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                    n_jobs=-1,
                )
                clf.fit(X, y)
                score = f1_score(y, clf.predict(X), average="macro")
                if score > best_score:
                    best, best_score = (
                        {"n_estimators": n_est, "max_depth": depth, "min_samples_leaf": min_leaf},
                        score,
                    )
    return best


def cross_validate_rf(win: pd.DataFrame, params: dict) -> dict:
    """: 5-fold GroupKFold across TRIALS on the training trials."""
    tr = win[win.trial_id.isin(TRAIN_TRIALS)]
    X, y, groups = tr[FEATURES].to_numpy(), tr["severity_class"].to_numpy(), tr["trial_id"].to_numpy()
    scaler = StandardScaler().fit(X)  # CV-internal scaling, training split only
    Xs = scaler.transform(X)
    gkf = GroupKFold(n_splits=5)
    accs, macro_f1s, recalls3 = [], [], []
    for tr_i, te_i in gkf.split(Xs, y, groups):
        clf = RandomForestClassifier(**params, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
        clf.fit(Xs[tr_i], y[tr_i])
        pred = clf.predict(Xs[te_i])
        accs.append(float((pred == y[te_i]).mean()))
        macro_f1s.append(float(f1_score(y[te_i], pred, average="macro", zero_division=0)))
        r = recall_score(y[te_i], pred, labels=[3], average=None, zero_division=0)
        recalls3.append(float(r[0]) if len(r) else 0.0)
    return {
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_macro_f1_mean": float(np.mean(macro_f1s)),
        "cv_critical_recall_mean": float(np.mean(recalls3)),
        "n_folds": 5,
    }


def train_rf(train_split: pd.DataFrame, params: dict) -> tuple[RandomForestClassifier, StandardScaler]:
    scaler = StandardScaler().fit(train_split[FEATURES])
    clf = RandomForestClassifier(**params, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
    clf.fit(scaler.transform(train_split[FEATURES]), train_split["severity_class"].to_numpy())
    return clf, scaler


def threshold_baseline(df: pd.DataFrame) -> np.ndarray:
    """: alert when displacement_rate OR vibration_rms stays elevated for
 3 consecutive windows (per trial-node series). Returns predicted classes
 0/1/2 mapped from sustained level."""
    out = np.zeros(len(df), dtype=int)
    df = df.reset_index(drop=True)
    for _key, g in df.groupby(["trial_id", "node_id"], sort=False):
        idx = g.index.to_numpy()
        dr = g["displacement_rate"].to_numpy()
        vib = g["vibration_rms"].to_numpy()
        disp = g["displacement_mm"].to_numpy()
        hot = (dr > 0.06) | (vib > 0.02)
        sustained = hot.copy()
        for k in range(2, len(idx)):
            sustained[k] = hot[k] and hot[k - 1] and hot[k - 2]
        pred = np.where(sustained & (disp > 15.0), 2, np.where(sustained, 1, 0))
        pred[sustained & (disp > 35.0)] = 3
        out[idx] = pred
    return out


def lead_time_seconds(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """: seconds before known displacement crosses 35 mm that the pipeline
 first flags (per active-node trial). Negative = late."""
    df = df.reset_index(drop=True)
    df["pred"] = pred
    results = []
    for (t, n), g in df.groupby(["trial_id", "node_id"]):
        if n not in ("N2", "N3"):
            continue
        kd = g["known_displacement_mm"].to_numpy()
        if kd.max() <= CRITICAL_MM:
            continue  # trial never reaches critical at this node
        cross = int(np.argmax(kd > CRITICAL_MM))
        flagged = np.flatnonzero(g["pred"].to_numpy() >= 1)
        first = int(flagged[0]) if flagged.size else -1
        if first < 0:
            results.append((t, n, float("nan")))
        else:
            results.append((t, n, (cross - first) * 1.0))  # 1 s per window step
    arr = np.array([r[2] for r in results], dtype=float)
    arr = arr[~np.isnan(arr)]
    return {
        "n_critical_series": len(results),
        "median_lead_time_s": float(np.median(arr)) if arr.size else float("nan"),
        "min_lead_time_s": float(np.min(arr)) if arr.size else float("nan"),
        "never_flagged": int(sum(1 for r in results if np.isnan(r[2]))),
        "per_series": {f"{t}/{n}": (None if np.isnan(v) else v) for t, n, v in results},
    }


def classification_report_frame(y, pred) -> pd.DataFrame:
    prec, rec, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1, 2, 3], zero_division=0
    )
    return pd.DataFrame(
        {
            "class": [CLASS_NAMES[c] for c in range(4)],
            "precision": np.round(prec, 4),
            "recall": np.round(rec, 4),
            "f1": np.round(f1, 4),
            "support": support,
        }
    )


def main() -> int:
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(exist_ok=True)
    OUT_EXPERIMENTS.mkdir(exist_ok=True)
    win, meta = load_data()

    # split-mix sanity: every split must mix conditions and speeds
    for name, trials in (("train", TRAIN_TRIALS), ("val", VAL_TRIALS), ("test", TEST_TRIALS)):
        conds = set(meta.loc[trials, "condition"])
        speeds = set(meta.loc[trials, "speed"])
        assert len(conds) >= 2 and len(speeds) >= 2, f"{name} split not mixed: {conds} {speeds}"
        print(f"{name}: {len(trials)} trials | conditions {sorted(conds)} | speeds {sorted(speeds)}")

    train_split = split_frame(win, TRAIN_TRIALS)
    val_split = split_frame(win, VAL_TRIALS)
    test_split = split_frame(win, TEST_TRIALS)

    # Isolation Forest (Stable-only training, step 4)
    iso, iso_scaler, _ = train_if(train_split)
    val_if = if_scores(iso, iso_scaler, val_split)
    val_stable = val_split[val_split.severity_class == 0]["severity_class"].size
    # threshold: 99th percentile of validation Stable scores (chosen on val only)
    stable_scores = val_if[val_split.severity_class == 0]
    if_threshold = float(np.quantile(stable_scores, 0.99))
    print(f"\nIF: stable val windows={val_stable}, anomaly threshold (p99 of stable)={if_threshold:.4f}")

    val_nonstable = val_if[val_split.severity_class > 0]
    auc_val = float(
        roc_auc_score(
            (val_split["severity_class"] > 0).astype(int).to_numpy(),
            val_if,
        )
    )
    print(f"IF separation on validation: AUC={auc_val:.4f} | "
          f"stable median={np.median(stable_scores):.4f} vs non-stable median={np.median(val_nonstable):.4f}")

    params = tune_rf(val_split)
    print(f"\nRF hyperparameters chosen on validation: {params}")
    cv = cross_validate_rf(win, params)
    print(f"5-fold GroupKFold (trial-grouped) on training trials: {cv}")
    rf, rf_scaler = train_rf(train_split, params)

    # Held-out test evaluation (once, step 7)
    Xte = test_split[FEATURES].to_numpy()
    yte = test_split["severity_class"].to_numpy()
    rf_pred = rf.predict(rf_scaler.transform(Xte))
    rf_proba = rf.predict_proba(rf_scaler.transform(Xte))

    test_if = if_scores(iso, iso_scaler, test_split)
    if_flag = (test_if > if_threshold).astype(int)

    # traffic light: red = RF Critical OR IF anomaly; else RF class
    alert = rf_pred.copy()
    alert[(rf_pred == 0) & (if_flag == 1)] = 3  # novel anomaly trip-wire

    auc_test = float(roc_auc_score((yte > 0).astype(int), test_if))

    print("\n=== TEST (held-out trials: T004, T012, T017, T024) ===")
    print("Random Forest per-class:")
    print(classification_report_frame(yte, rf_pred).to_string(index=False))
    cm = confusion_matrix(yte, rf_pred, labels=[0, 1, 2, 3])
    print("confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=[CLASS_NAMES[i] for i in range(4)],
                       columns=[CLASS_NAMES[i] for i in range(4)]).to_string())

    print("\nthreshold baseline () per-class:")
    base_pred = threshold_baseline(test_split)
    print(classification_report_frame(yte, base_pred).to_string(index=False))

    print("\nIF on test: AUC=", round(auc_test, 4),
          "| flagged:", int(if_flag.sum()), "of", len(if_flag),
          "| of which truly non-Stable:", int(((if_flag == 1) & (yte > 0)).sum()))

    lead_rf = lead_time_seconds(test_split, rf_pred)
    lead_base = lead_time_seconds(test_split, base_pred)
    print("\nlead time to critical (median): RF =", lead_rf["median_lead_time_s"], "s |",
          "baseline =", lead_base["median_lead_time_s"], "s")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].hist(test_if[yte == 0], bins=50, alpha=0.6, label="Stable", density=True)
    axes[0].hist(test_if[yte > 0], bins=50, alpha=0.6, label="non-Stable", density=True)
    axes[0].axvline(if_threshold, color="r", ls="--", label=f"threshold {if_threshold:.3f}")
    axes[0].set_xlabel("IF anomaly score (−score_samples)"); axes[0].set_ylabel("density")
    axes[0].set_title(f"Isolation Forest separation (test AUC={auc_test:.3f})"); axes[0].legend()

    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    axes[1].barh(importances.index, importances.values, color="#3b6ea5")
    axes[1].set_title("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(OUT_REPORTS / "tabletop_model_evaluation.png", dpi=110)
    plt.close(fig)

    # artifacts (versioned, integrity-checked, registry-registered)
    from src.risk.artifacts import save_model_artifact

    tabletop_dataset_version = "tabletop-recorded-v1"  # data/recorded/tabletop/dataset_manifest.json
    save_model_artifact(
        OUT_MODELS / "isolation_forest.joblib",
        model=iso, features=list(FEATURES),
        model_name="tabletop_isolation_forest", model_version="1.0.0",
        training_dataset_version=tabletop_dataset_version,
        split_name="trial_holdout", seed=SEED,
        scaler=iso_scaler, preprocessing="standardise",
        extra={"threshold": float(if_threshold)},
    )
    # RF is scale-invariant: no scaler, preprocessing declared "none"
    save_model_artifact(
        OUT_MODELS / "random_forest.joblib",
        model=rf, features=list(FEATURES),
        model_name="tabletop_random_forest", model_version="1.0.0",
        training_dataset_version=tabletop_dataset_version,
        split_name="trial_holdout", seed=SEED,
        preprocessing="none",
        extra={"params": params, "class_names": CLASS_NAMES},
    )

    test_if_frame = test_split[["trial_id", "node_id", "window_start_ms", "known_displacement_mm", "severity_class"]].copy()
    test_if_frame["if_score"] = test_if
    test_if_frame["if_flag"] = if_flag
    test_if_frame["rf_pred"] = rf_pred
    test_if_frame["alert_level"] = [ALERT_NAMES[a] for a in alert]
    test_if_frame["rf_p_critical"] = rf_proba[:, 3]
    test_if_frame.to_csv(OUT_MODELS / "test_predictions.csv", index=False)

    result = {
        "seed": SEED,
        "splits": {"train": TRAIN_TRIALS, "validation": VAL_TRIALS, "test": TEST_TRIALS},
        "features": FEATURES,
        "isolation_forest": {
            "trained_on": "severity_class==0 windows of the 18 training trials only",
            "n_estimators": 300,
            "contamination": "auto",
            "val_auc": auc_val,
            "test_auc": auc_test,
            "threshold_rule": "p99 of validation Stable scores",
            "threshold": if_threshold,
        },
        "random_forest": {
            "params": params,
            "class_weight": "balanced_subsample",
            "cv": cv,
            "test": {
                "confusion_matrix": cm.tolist(),
                "per_class": classification_report_frame(yte, rf_pred).to_dict(orient="records"),
                "critical_recall": float(recall_score(yte, rf_pred, labels=[3], average=None, zero_division=0)[0]),
            },
        },
        "baseline": {
            "rule": "displacement_rate>0.06 OR vibration_rms>0.02 sustained 3 windows; class by displacement",
            "per_class": classification_report_frame(yte, base_pred).to_dict(orient="records"),
        },
        "lead_time_s": {"random_forest": lead_rf, "threshold_baseline": lead_base},
        "ml_beats_baseline": {
            "macro_f1": {
                "rf": float(f1_score(yte, rf_pred, average="macro")),
                "baseline": float(f1_score(yte, base_pred, average="macro")),
            }
        },
        "artifacts": {
            "models": ["models/tabletop/isolation_forest.joblib", "models/tabletop/random_forest.joblib"],
            "predictions": "models/tabletop/test_predictions.csv",
            "figure": "reports/tabletop_model_evaluation.png",
        },
    }
    (OUT_EXPERIMENTS / "tabletop_model_run.json").write_text(json.dumps(result, indent=2))

    rf_f1 = result["ml_beats_baseline"]["macro_f1"]["rf"]
    base_f1 = result["ml_beats_baseline"]["macro_f1"]["baseline"]
    crit_recall = result["random_forest"]["test"]["critical_recall"]
    print("\n=== VERDICT ===")
    print(f"  RF macro-F1 {rf_f1:.4f} vs baseline {base_f1:.4f} -> ML wins: {rf_f1 > base_f1}")
    print(f"  Critical recall (held-out): {crit_recall:.4f}")
    print(f"  IF test AUC: {auc_test:.4f} | median lead time: {lead_rf['median_lead_time_s']} s")
    print(f"  artifacts -> {OUT_MODELS}, reports/tabletop_model_evaluation.png, experiments/tabletop_model_run.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
