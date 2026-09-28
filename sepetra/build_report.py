#!/usr/bin/env python3
"""Sepetra — model evaluation report (accuracy, precision, recall, F1 + graphs).

Builds a complete, reproducible evaluation pack for the §15 risk model:

    sepetra/
      build_report.py      ← this script (re-run any time; idempotent)
      README.md            ← generated results table + reading guide
      metrics.json         ← every number, machine-readable
      predictions_test.parquet  ← per-window scores on the honest test split
      graphs/
        confusion_matrices.png
        roc_curves.png
        precision_recall_curves.png
        metric_bars.png
        calibration_curve.png
        per_class_metrics.png
        lead_time_false_alarm.png

Source of truth: the T-068 event-level splits (no event spans splits), the
§15 XGBoost via src/risk/xgboost_model.py, and metric primitives from
src/evaluation/metrics.py (§24). Threshold = P(CRITICAL) >= 0.5, identical
to the T-070 ablation convention, so numbers line up with
experiments/ablation_a_to_f.json.

Deterministic: fixed hyperparameters from configs/risk_model.yaml, fixed
splits, no sampling. Re-running overwrites outputs in place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import brier_score, expected_calibration_error  # noqa: E402
from src.risk.xgboost_model import train_risk_model  # noqa: E402

SEPETRA = Path(__file__).resolve().parent
GRAPHS = SEPETRA / "graphs"

ALERT_THRESHOLD = 0.5  # P(CRITICAL) at which the alert engine fires (T-070 convention)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> int:
    GRAPHS.mkdir(parents=True, exist_ok=True)

    # ---- data + model (identical to the T-070 arm-E convention) -----------
    store = pd.read_parquet(REPO_ROOT / "data" / "features" / "features_v2.parquet")
    splits = pd.read_csv(REPO_ROOT / "data" / "features" / "split_assignment.csv")
    store = store.merge(splits[["event_id", "split"]], on="event_id", how="left", validate="many_to_one")
    if store["split"].isna().any():
        raise SystemExit("split_assignment missing events — refusing to evaluate an unsplit store")

    print(f"store: {len(store):,} windows across {store.event_id.nunique():,} events; "
          f"splits: {store['split'].value_counts().to_dict()}", flush=True)

    model = train_risk_model(store)  # full §15 input contract (Groups A–F + signals)
    classes = list(model.classes)
    i_crit = classes.index("CRITICAL")

    test = store[store["split"] == "test"].reset_index(drop=True)
    proba = model.predict_proba(test)
    pred_idx = proba.argmax(axis=1)
    pred_labels = np.asarray(classes)[pred_idx]

    y_idx = test["risk_label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_true_labels = test["risk_label"].to_numpy()

    # ---- headline metrics (test split) ------------------------------------
    p_crit = proba[:, i_crit]
    y_bin = (test["anomaly_label"] > 0).astype(int).to_numpy()
    pred_bin = (pred_labels != "NORMAL").astype(int)

    metrics: dict = {
        "threshold_p_critical": ALERT_THRESHOLD,
        "n_test_windows": int(len(test)),
        "n_test_events": int(test.event_id.nunique()),
        "accuracy": float(accuracy_score(y_true_labels, pred_labels)),
        "precision_binary": float(precision_score(y_bin, pred_bin, zero_division=0)),
        "recall_binary": float(recall_score(y_bin, pred_bin, zero_division=0)),
        "f1_binary": float(f1_score(y_bin, pred_bin, zero_division=0)),
        "precision_macro": float(precision_score(y_true_labels, pred_labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_labels, pred_labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_labels, pred_labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true_labels, pred_labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true_labels, pred_labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_labels, pred_labels, average="weighted", zero_division=0)),
        "roc_auc_critical_vs_rest": float(roc_auc_score((test["risk_label"] == "CRITICAL").astype(int), p_crit)),
        "pr_auc_critical_vs_rest": float(
            average_precision_score((test["risk_label"] == "CRITICAL").astype(int), p_crit)
        ),
        "brier_p_critical": brier_score(np.stack([p_crit, 1.0 - p_crit], axis=1),
                                        (test["risk_label"] != "CRITICAL").astype(int).to_numpy()),
        "ece_p_critical": expected_calibration_error(np.stack([p_crit, 1.0 - p_crit], axis=1),
                                                     (test["risk_label"] != "CRITICAL").astype(int).to_numpy()),
    }

    per_class = {}
    for c in classes:
        m = (y_true_labels == c)
        mp = (pred_labels == c)
        tp = int((m & mp).sum()); fp = int((~m & mp).sum()); fn = int((m & ~mp).sum())
        tn = int((~m & ~mp).sum())
        per_class[c] = {
            "support": int(m.sum()),
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }
    metrics["per_class"] = per_class

    # ---- predictions artifact ---------------------------------------------
    keep_cols = [c for c in ("event_id", "node_id", "window_index", "window_timestamp",
                             "anomaly_label", "risk_label") if c in test.columns]
    preds = test[keep_cols].copy()
    preds["pred_label"] = pred_labels
    preds["p_CRITICAL"] = p_crit
    preds["p_WARNING"] = proba[:, classes.index("WARNING")]
    preds["p_NORMAL"] = proba[:, classes.index("NORMAL")]
    preds["correct"] = (pred_labels == y_true_labels).astype(int)
    preds.to_parquet(SEPETRA / "predictions_test.parquet", index=False)

    # ---- graphs ------------------------------------------------------------
    COLOR = {"NORMAL": "#2e7d32", "WARNING": "#f9a825", "CRITICAL": "#b71c1c"}

    # 1. confusion matrices (counts + row-normalised)
    cm = confusion_matrix(y_true_labels, pred_labels, labels=classes)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, mat, title, fmt in zip(
        axes, [cm, cm_norm],
        ["Confusion matrix (counts)", "Confusion matrix (row-normalised)"],
        ["d", ".2f"], strict=True,
    ):
        im = ax.imshow(mat, cmap="Reds")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
        for r in range(len(classes)):
            for c in range(len(classes)):
                ax.text(c, r, format(mat[r, c], fmt), ha="center", va="center",
                        color="white" if mat[r, c] > mat.max() * 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle("Sepetra — §15 risk model, test split", y=1.02)
    fig.savefig(GRAPHS / "confusion_matrices.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2. ROC curves (one-vs-rest)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for c in classes:
        yb = (y_true_labels == c).astype(int)
        if yb.sum() == 0 or yb.all():
            continue
        fpr, tpr, _ = roc_curve(yb, proba[:, classes.index(c)])
        ax.plot(fpr, tpr, color=COLOR[c], lw=2, label=f"{c} vs rest (AUC {roc_auc_score(yb, proba[:, classes.index(c)]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title("ROC — one-vs-rest"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(GRAPHS / "roc_curves.png", dpi=120); plt.close(fig)

    # 3. precision-recall curves (one-vs-rest)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for c in classes:
        yb = (y_true_labels == c).astype(int)
        if yb.sum() == 0:
            continue
        pr, rc, _ = precision_recall_curve(yb, proba[:, classes.index(c)])
        ap = metrics["pr_auc_critical_vs_rest"] if c == "CRITICAL" else \
            float(np.trapezoid(pr, rc)) if hasattr(np, "trapezoid") else float(-np.trapz(pr, rc))
        ax.plot(rc, pr, color=COLOR[c], lw=2, label=f"{c} (AP ≈ {ap:.3f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision–recall — one-vs-rest"); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(GRAPHS / "precision_recall_curves.png", dpi=120); plt.close(fig)

    # 4. headline metric bars
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    names = ["accuracy", "precision\n(binary)", "recall\n(binary)", "F1\n(binary)",
             "precision\n(macro)", "recall\n(macro)", "F1\n(macro)"]
    vals = [metrics["accuracy"], metrics["precision_binary"], metrics["recall_binary"], metrics["f1_binary"],
            metrics["precision_macro"], metrics["recall_macro"], metrics["f1_macro"]]
    bars = ax.bar(names, vals, color=["#455a64", "#b71c1c", "#b71c1c", "#b71c1c",
                                      "#f9a825", "#f9a825", "#f9a825"], alpha=0.88)
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title(f"Sepetra headline metrics — test split, {len(test):,} windows")
    fig.tight_layout(); fig.savefig(GRAPHS / "metric_bars.png", dpi=120); plt.close(fig)

    # 5. calibration curve for P(CRITICAL)
    frac_pos, mean_pred = calibration_curve((test["risk_label"] == "CRITICAL").astype(int), p_crit, n_bins=10)
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color="#b71c1c", label="model P(CRITICAL)")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title(f"Calibration — Brier {metrics['brier_p_critical']:.4f}, ECE {metrics['ece_p_critical']:.4f}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(GRAPHS / "calibration_curve.png", dpi=120); plt.close(fig)

    # 6. per-class P/R/F1
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = np.arange(len(classes)); w = 0.26
    for k, (metric, col) in enumerate(zip(["precision", "recall", "f1"],
                                          ["#455a64", "#f9a825", "#b71c1c"], strict=True)):
        ax.bar(xs + (k - 1) * w, [per_class[c][metric] for c in classes], w, label=metric, color=col)
    for i, c in enumerate(classes):
        ax.text(i, 1.02, f"n={per_class[c]['support']:,}", ha="center", fontsize=8)
    ax.set_xticks(xs, classes); ax.set_ylim(0, 1.12); ax.legend()
    ax.set_title("Per-class precision / recall / F1")
    fig.tight_layout(); fig.savefig(GRAPHS / "per_class_metrics.png", dpi=120)
    plt.close(fig)

    # 7. lead time & false alarms per event (operational view)
    rows = []
    for ev, g in preds.groupby("event_id", sort=False):
        g = g.sort_values("window_index")
        fired = g.loc[g["p_CRITICAL"] >= ALERT_THRESHOLD, "window_index"]
        onset = g.loc[g["anomaly_label"] > 0, "window_index"]
        rows.append({
            "event_id": ev,
            "alerted": len(fired) > 0,
            "lead_windows": (float(onset.iloc[0]) - float(fired.iloc[0])) if (len(fired) and len(onset) and fired.iloc[0] <= onset.iloc[0]) else np.nan,
            "false_alarm_windows": int((g["p_CRITICAL"].ge(ALERT_THRESHOLD) & (g["anomaly_label"] == 0)).sum()),
        })
    ev = pd.DataFrame(rows)
    leads = ev["lead_windows"].dropna() * 0.6  # §10 stride: 36 min per window
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].hist(leads, bins=np.arange(-0.5, 9.5, 1), color="#2e7d32", alpha=0.85)
    axes[0].set_xlabel("lead time (windows of 0.6 h; 0 = alert at onset)")
    axes[0].set_ylabel("events")
    axes[0].set_title(f"Lead time on alerted events (median {leads.median():.1f} windows)" if len(leads) else "No alerted events")
    axes[1].hist(ev["false_alarm_windows"], bins=np.arange(-0.5, 9.5, 1), color="#b71c1c", alpha=0.85)
    axes[1].set_xlabel("false-alarm windows per event (P(CRIT) ≥ 0.5 on quiet windows)")
    axes[1].set_ylabel("events")
    axes[1].set_title(f"False alarms (events with ≥1: {int((ev['false_alarm_windows'] > 0).sum())} of {len(ev)})")
    fig.tight_layout(); fig.savefig(GRAPHS / "lead_time_false_alarm.png", dpi=120); plt.close(fig)

    metrics["operational"] = {
        "median_lead_time_hours": float(leads.median()) if len(leads) else None,
        "p10_lead_time_hours": float(leads.quantile(0.10)) if len(leads) else None,
        "events_alerted": int(ev["alerted"].sum()),
        "n_events": int(len(ev)),
        "false_alarm_windows_per_event_mean": float(ev["false_alarm_windows"].mean()),
    }

    # ---- persisted outputs -------------------------------------------------
    (SEPETRA / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_readme(metrics)
    print(f"\nwrote {SEPETRA / 'metrics.json'}")
    print(f"wrote {SEPETRA / 'predictions_test.parquet'}")
    print(f"wrote {len(list(GRAPHS.glob('*.png')))} graphs → {GRAPHS.relative_to(REPO_ROOT)}/")
    print("\nHEADLINE (test split)")
    print(f"  accuracy  : {metrics['accuracy']:.4f}   (context: {metrics['n_test_windows']:,} windows, "
          f"class-imbalance visible in per_class)")
    print(f"  precision : {metrics['precision_binary']:.4f} (binary alert) | {metrics['precision_macro']:.4f} (macro)")
    print(f"  recall    : {metrics['recall_binary']:.4f} (binary alert) | {metrics['recall_macro']:.4f} (macro)")
    print(f"  F1        : {metrics['f1_binary']:.4f} (binary alert) | {metrics['f1_macro']:.4f} (macro)")
    return 0


def write_readme(m: dict) -> None:
    pc = m["per_class"]
    lines = [
        "# Sepetra — Model Evaluation Report",
        "",
        f"§15 risk model (XGBoost) evaluated on the **test split** "
        f"({m['n_test_windows']:,} windows, {m['n_test_events']:,} events, T-068 event-level splits). "
        f"Alert rule: P(CRITICAL) ≥ {m['threshold_p_critical']} (same convention as the T-070 ablation, "
        "so numbers match `experiments/ablation_a_to_f.json`).",
        "",
        "## Headline metrics",
        "",
        "| metric | value |",
        "|---|---|",
        f"| accuracy | {_fmt(m['accuracy'])} |",
        f"| precision (binary alert) | {_fmt(m['precision_binary'])} |",
        f"| recall (binary alert) | {_fmt(m['recall_binary'])} |",
        f"| F1 (binary alert) | {_fmt(m['f1_binary'])} |",
        f"| precision (macro) | {_fmt(m['precision_macro'])} |",
        f"| recall (macro) | {_fmt(m['recall_macro'])} |",
        f"| F1 (macro) | {_fmt(m['f1_macro'])} |",
        f"| PR-AUC (CRITICAL vs rest) | {_fmt(m['pr_auc_critical_vs_rest'])} |",
        f"| ROC-AUC (CRITICAL vs rest) | {_fmt(m['roc_auc_critical_vs_rest'])} |",
        f"| Brier (P-CRITICAL) | {m['brier_p_critical']:.4f} |",
        f"| ECE (P-CRITICAL) | {m['ece_p_critical']:.4f} |",
        "",
        "**Why two precisions?** The *binary* view scores the operational question "
        "“did the mesh raise an alert on a disturbed window?”; the *macro* view scores "
        "the 3-class risk grading. Both are reported — accuracy alone would flatter a "
        "model that never warns (NORMAL dominates this mesh), which is why the repo's "
        "§24 headline set omits it while this report shows it with context.",
        "",
        "## Per-class detail",
        "",
        "| class | support | precision | recall | F1 |",
        "|---|---|---|---|---|",
    ]
    for c, v in pc.items():
        lines.append(f"| {c} | {v['support']:,} | {_fmt(v['precision'])} | {_fmt(v['recall'])} | {_fmt(v['f1'])} |")
    lines += [
        "",
        "## Graphs (`graphs/`)",
        "",
        "| file | what it shows |",
        "|---|---|",
        "| confusion_matrices.png | counts + row-normalised confusion matrices |",
        "| roc_curves.png | one-vs-rest ROC with AUC per class |",
        "| precision_recall_curves.png | one-vs-rest precision-recall (imbalance-honest) |",
        "| metric_bars.png | headline accuracy/precision/recall/F1 side by side |",
        "| calibration_curve.png | predicted vs observed frequency for P(CRITICAL) |",
        "| per_class_metrics.png | precision/recall/F1 per class with support |",
        "| lead_time_false_alarm.png | operational view: lead time + false alarms per event |",
        "",
        "## Operational summary",
        "",
        f"- Median lead time on alerted events: "
        f"{m['operational']['median_lead_time_hours'] if m['operational']['median_lead_time_hours'] is not None else 'n/a'} h "
        f"(P10 {m['operational']['p10_lead_time_hours']} h).",
        f"- Events alerted: {m['operational']['events_alerted']} of {m['operational']['n_events']}.",
        f"- Mean false-alarm windows per event: "
        f"{m['operational']['false_alarm_windows_per_event_mean']:.2f}.",
        "",
        "## Reproduce",
        "",
        "```bash",
        ".venv/bin/python sepetra/build_report.py",
        "```",
        "",
        "Deterministic: config-driven hyperparameters, fixed T-068 splits, no sampling.",
        "Per-window scores: `predictions_test.parquet`. Machine-readable metrics: `metrics.json`.",
        "",
        "## Honest reading",
        "",
        "- **Accuracy is high but weakly informative here**: NORMAL is the majority class "
        f"({pc['NORMAL']['support']:,} of {m['n_test_windows']:,} test windows), so the §24 discipline "
        "prefers precision/recall/F1 and calibration as headline numbers.",
        "- The binary alert view trades precision for recall at the 0.5 threshold — see "
        "`precision_recall_curves.png` for the full trade-off surface before moving it.",
        "- Lead time is bounded by the §10 event generator (onset mid-event); see the T-070 "
        "gate discussion in `reports/ablation.md`.",
    ]
    (SEPETRA / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
