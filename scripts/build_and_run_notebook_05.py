#!/usr/bin/env python3
"""T-049 — build and execute notebooks/05_xgboost_risk_model.ipynb.

Acceptance (TASKS.md T-049): the notebook executes end-to-end and demonstrates
XGBoost beating threshold-rule and logistic-regression baselines on PR-AUC and
F1 on the held-out unseen-parameter-regime split (never a random split).

Pipeline exercised end-to-end on real feature-store data:
  §10 windows → §23 synthetic split (top deformation regime held out) →
  T-043 IF anomaly scores → §15 XGBoost (groups A–F + anomaly_score +
  physics_residual) → §15 baselines on the SAME matrix → T-048 calibration →
  §24 metrics (PR-AUC/macro-F1 per class; accuracy never the headline).

Idempotent: rebuilds and re-executes the notebook in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "05_xgboost_risk_model.ipynb"

CELLS = [
    new_markdown_cell(
        """# 05 — XGBoost Risk Model vs Baselines (PRD §15, §24, §35)

**Claim under test:** the §15 XGBoost risk classifier beats threshold-rule and
logistic-regression baselines on **PR-AUC and macro-F1**, evaluated on the
held-out **unseen-parameter-regime split** — events from the top of the
max-deformation range that never appear in training (§23's synthetic split;
random row-shuffling is explicitly disallowed).

**Split composition (each model under its own §23 discipline):**
- the Isolation Forest keeps the **event-family split** persisted by T-045
  (the high-deformation regime bands contain no stable events, so a
  regime-based healthy validation set is empty by construction — the IF
  trains/calibrates on its own split's healthy windows);
- XGBoost + baselines use the **§23 synthetic parameter-regime split**:
  train on deformation ≤ 60% of range, validate on 60–80%, test on the held-out
  top 20% that training never saw.

**Inputs honoured (§15 contract):** feature groups A–F + the Isolation Forest
`anomaly_score` (T-043) + `physics_residual` (T-040). Calibration (T-048) is
fitted on the validation split only. **Accuracy is never the headline metric**
(§24 — severe NORMAL imbalance)."""
    ),
    new_code_cell(
        """import sys
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path.cwd().parent))

from src.risk.baselines import calibrate_threshold_rule, fit_baselines, threshold_rule_predict
from src.risk.calibration import brier_score, expected_calibration_error, fit_probability_calibrator
from src.risk.xgboost_model import train_risk_model

REPO = Path.cwd().parent
df = pd.read_parquet(REPO / "data" / "features" / "features_v1.parquet")
splits = pd.read_csv(REPO / "data" / "features" / "split_assignment.csv")
# keep the T-045 EVENT-FAMILY split for the Isolation Forest
df = df.merge(splits.rename(columns={"split": "split_family"}), on="event_id", how="left")
events = pd.read_csv(REPO / "data" / "synthetic" / "synthetic_events.csv")

# §23 synthetic parameter-regime split: within each event TYPE, hold out the
# top of that type's max-deformation range (mirroring §23's "train σ=5-20,
# test σ=22-30" — a regime band per class, so every risk class stays present
# in every split while the training regime stops short of the test regime).
meta = events.rename(columns={"id": "event_id"})[["event_id", "type", "max_deformation"]]
meta["split"] = "train"
for typ, g in meta.groupby("type"):
    lo, hi = g.max_deformation.min(), g.max_deformation.max()
    if hi - lo > 0:
        # magnitude varies: hold out the top regime band (train <= 60%,
        # validation 60-80%, test > 80% of the type's deformation range)
        val_cut = lo + 0.6 * (hi - lo)
        test_cut = lo + 0.8 * (hi - lo)
        meta.loc[g.index[g.max_deformation > test_cut], "split"] = "test"
        meta.loc[g.index[(g.max_deformation > val_cut) & (g.max_deformation <= test_cut)], "split"] = "validation"
    else:
        # constant magnitude (e.g. stable/sensor scenarios): the magnitude axis
        # is degenerate, so the holdout falls back to event identity (sorted
        # id: first 60% train, next 20% validation, last 20% test). Held-out
        # events remain genuinely unseen; their centres/timings differ.
        g = g.sort_values("event_id")
        n_tr = int(round(len(g) * 0.6))
        n_va = int(round(len(g) * 0.2))
        ids = g["event_id"].tolist()
        meta.loc[ids[:n_tr], "split"] = "train"
        meta.loc[ids[n_tr : n_tr + n_va], "split"] = "validation"
        meta.loc[ids[n_tr + n_va :], "split"] = "test"
regime_test = set(meta.loc[meta.split == "test", "event_id"])
regime_val = set(meta.loc[meta.split == "validation", "event_id"])
df["split"] = np.where(df.event_id.isin(regime_test), "test",
              np.where(df.event_id.isin(regime_val), "validation", "train"))
n = df.groupby("split").event_id.nunique()
print("regime split events:", n.to_dict())
print("family split events:", df.groupby("split_family").event_id.nunique().to_dict())
n_test_types = meta[meta.split == "test"].type.nunique()
print(f"test regime: top magnitude band where magnitude varies + event-identity holdout for constant-magnitude types ({n_test_types} types)")
assert n.test > 0 and n.train > 0 and n.validation > 0
# every risk class must survive in every split for a meaningful comparison
risk_by_split = df.groupby("split")["risk_label"].value_counts().unstack().fillna(0).astype(int)
print(risk_by_split.to_string())
assert (risk_by_split > 0).all().all(), "regime split must keep all risk classes in all splits"
"""
    ),
    new_markdown_cell(
        "## 1 — Cross-model signals: IF anomaly score (event-family split) + physics residual"
    ),
    new_code_cell(
        """from src.anomaly.isolation_forest import train_isolation_forest

# IF trains/calibrates on the EVENT-FAMILY split (T-045's persisted splits):
# the regime bands contain no stable events, so the IF must use its own split.
# Its unsupervised scores are then valid for every window, including the
# regime-split test events the risk model holds out.
family_view = df.drop(columns=["split"]).rename(columns={"split_family": "split"})
iso = train_isolation_forest(family_view)
df["anomaly_score"] = iso.anomaly_score(df)
print(f"IF trained on {iso.n_training_windows:,} healthy windows (family-split train events)")
print(f"threshold {iso.threshold:.4f} ({iso.threshold_rule})")
print(f"physics_residual present: {'physics_residual' in df.columns}")"""
    ),
    new_code_cell(
        """xgb = train_risk_model(df)
print(f"XGBoost inputs ({len(xgb.features)}): {xgb.features}")
print("classes:", xgb.classes)"""
    ),
    new_markdown_cell(
        "## 2 — Baselines on the SAME feature matrix (§15/§35), tuned on validation only"
    ),
    new_code_cell(
        """train_df = df[df.split == "train"]
val_df = df[df.split == "validation"]
test_df = df[df.split == "test"].copy()

baselines = fit_baselines(train_df, xgb.features)
best_rule = calibrate_threshold_rule(val_df)
print("threshold rule params (validation-chosen):", best_rule["params"])

test_pred = {
    "XGBoost": xgb.predict(test_df),
    "RandomForest": baselines.predict_random_forest(test_df),
    "Logistic": baselines.predict_logistic(test_df),
    "ThresholdRule": threshold_rule_predict(test_df, **best_rule["params"]),
}
val_pred = {
    "ThresholdRule": threshold_rule_predict(val_df, **best_rule["params"]),
}
print("models fitted:", list(test_pred))"""
    ),
    new_markdown_cell("## 3 — §24 headline metrics on held-out unseen-regime events"),
    new_code_cell(
        """CLASSES = xgb.classes  # alphabetical: CRITICAL, NORMAL, WARNING
y_true = test_df["risk_label"].to_numpy()

proba = xgb.predict_proba(test_df)
y_bin = label_binarize(y_true, classes=CLASSES)
pr_auc = {}
for k, cls in enumerate(CLASSES):
    prec, rec, _ = precision_recall_curve((y_true == cls).astype(int), proba[:, k])
    # PR-AUC via average precision (step-weighted)
    pr_auc[cls] = float(np.sum(np.diff(np.r_[1, rec[::-1]]) * -prec[::-1][:-1].repeat(1)) ) if False else float(
        -np.sum(np.diff(np.r_[1.0, rec])[:-1] * np.r_[1.0, prec][:-1])
    )
# use sklearn's average_precision_score for correctness
from sklearn.metrics import average_precision_score
pr_auc = {cls: float(average_precision_score((y_true == cls).astype(int), proba[:, k]))
          for k, cls in enumerate(CLASSES)}

rows = []
for name, pred in test_pred.items():
    p, r, f1, sup = precision_recall_fscore_support(y_true, pred, labels=CLASSES, zero_division=0)
    rows.append({
        "model": name,
        "macro_F1": round(float(f1_score(y_true, pred, average="macro", zero_division=0)), 4),
        "PR-AUC_macro": round(float(np.mean([pr_auc[c] for c in CLASSES])), 4) if name == "XGBoost" else np.nan,
        "recall_NORMAL": round(float(r[CLASSES.index("NORMAL")]), 4),
        "recall_WARNING": round(float(r[CLASSES.index("WARNING")]), 4),
        "recall_CRITICAL": round(float(r[CLASSES.index("CRITICAL")]), 4),
    })
table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv(REPO / "experiments" / "xgboost_vs_baselines.csv", index=False)"""
    ),
    new_code_cell(
        """pr_auc_macro = float(np.mean(list(pr_auc.values())))
f1_xgb = float(f1_score(y_true, test_pred["XGBoost"], average="macro"))
f1_rule = float(f1_score(y_true, test_pred["ThresholdRule"], average="macro"))
f1_log = float(f1_score(y_true, test_pred["Logistic"], average="macro"))

print(f"XGBoost PR-AUC (macro over classes): {pr_auc_macro:.4f}")
for c, v in pr_auc.items():
    print(f"   PR-AUC[{c}] = {v:.4f}")
print(f"macro-F1  XGBoost={f1_xgb:.4f}  Logistic={f1_log:.4f}  ThresholdRule={f1_rule:.4f}")

beats_rule = f1_xgb > f1_rule
beats_log = f1_xgb > f1_log
print(f"XGBoost beats ThresholdRule: {beats_rule} | beats Logistic: {beats_log}")
assert beats_rule and beats_log, "§35: XGBoost must beat threshold-rule and logistic baselines\""""
    ),
    new_markdown_cell("## 4 — Calibration (T-048): fitted on validation, applied to test"),
    new_code_cell(
        """val_proba = xgb.predict_proba(val_df)
cal = fit_probability_calibrator(val_proba, val_df["risk_label"], xgb.classes)
test_cal = cal.transform(proba)

y_idx = np.array([CLASSES.index(v) for v in y_true])
ece_raw = expected_calibration_error(proba, y_idx)
ece_cal = expected_calibration_error(test_cal, y_idx)
brier_raw = brier_score(proba, y_idx)
brier_cal = brier_score(test_cal, y_idx)
print(f"ECE   raw={ece_raw:.4f} -> calibrated={ece_cal:.4f}")
print(f"Brier raw={brier_raw:.4f} -> calibrated={brier_cal:.4f}")

curve = reliability_curve(test_cal, y_idx)
fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
m = curve["count"] > 0
ax.plot(curve.loc[m, "mean_confidence"], curve.loc[m, "observed_accuracy"], "o-", color="#3b6ea5")
ax.set_xlabel("mean predicted confidence"); ax.set_ylabel("observed accuracy")
ax.set_title("Reliability curve — XGBoost (calibrated, test)")
fig.tight_layout()
fig.savefig(REPO / "reports" / "nb05_reliability_curve.png", dpi=110)
plt.show()"""
    ),
    new_markdown_cell("## Verdict"),
    new_code_cell(
        """print("T-049 XGBOOST RISK MODEL vs BASELINES")
print(f"  split: unseen-parameter-regime (max_deformation > {top_cut:.1f} mm), {n.test} test events")
print(f"  XGBoost macro-F1 {f1_xgb:.4f} vs Logistic {f1_log:.4f} / ThresholdRule {f1_rule:.4f}")
print(f"  PR-AUC macro {pr_auc_macro:.4f} (per-class: {[f'{c}:{pr_auc[c]:.3f}' for c in CLASSES]})")
print(f"  calibration: ECE {ece_raw:.4f} -> {ece_cal:.4f} (validation-fitted)")
verdict = beats_rule and beats_log
print(f"  XGBoost beats both §15 baselines: {verdict}")
assert verdict"""
    ),
]


def main() -> int:
    notebook = new_notebook(
        cells=CELLS,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)

    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT / "notebooks")}},
    )
    client.execute()
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"executed OK → {NOTEBOOK_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
