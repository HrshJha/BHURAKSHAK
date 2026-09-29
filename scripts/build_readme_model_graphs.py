"""Build the README model plots from the one-time locked evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "reports/final_eval.json").read_text(encoding="utf-8"))
OUT = ROOT / "reports"


def risk_model_plot() -> None:
    models = RESULTS["risk_models"]
    names = {
        "tuned_xgboost": "Tuned XGBoost",
        "default_xgboost": "Default XGBoost",
        "logistic_regression": "Logistic regression",
        "threshold_rule": "Threshold rule",
    }
    metrics = (
        ("pr_auc_macro_ovr", "Macro PR-AUC", "#1D3557"),
        ("f1_macro", "Macro F1", "#457B9D"),
        ("recall_critical", "CRITICAL recall", "#E09F3E"),
    )
    positions = np.arange(len(names))
    offsets = np.linspace(-0.22, 0.22, len(metrics))
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    for offset, (key, label, color) in zip(offsets, metrics, strict=True):
        values = [models[name][key] for name in names]
        bars = ax.barh(positions + offset, values, height=0.19, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    ax.set_yticks(positions, [names[name] for name in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("Score")
    ax.set_title("Risk models · one-time locked synthetic evaluation")
    ax.grid(axis="x", color="#d9dee5", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, ncols=3)
    fig.savefig(OUT / "readme_risk_model_metrics.png", dpi=180, facecolor="white")
    plt.close(fig)


def confusion_matrix_plot() -> None:
    result = RESULTS["risk_models"]["tuned_xgboost"]
    counts = np.asarray(result["confusion_matrix_labels_normal_warning_critical"])
    row_totals = counts.sum(axis=1, keepdims=True)
    rates = counts / row_totals
    labels = ("NORMAL", "WARNING", "CRITICAL")
    fig, ax = plt.subplots(figsize=(6.4, 5.3), layout="constrained")
    image = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(
                col,
                row,
                f"{rates[row, col]:.1%}\n({counts[row, col]:,})",
                ha="center",
                va="center",
                color="white" if rates[row, col] > 0.52 else "#17212b",
                fontsize=10,
            )
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Tuned XGBoost · row-normalized confusion matrix")
    fig.colorbar(image, ax=ax, label="Share of true-class windows")
    fig.savefig(OUT / "readme_tuned_confusion_matrix.png", dpi=180, facecolor="white")
    plt.close(fig)


def anomaly_model_plot() -> None:
    models = RESULTS["anomaly_models"]
    names = ("default", "tuned")
    metrics = (
        ("pr_auc_anomaly", "Anomaly PR-AUC", "#1D3557"),
        ("recall_at_development_healthy_threshold", "Recall at development threshold", "#457B9D"),
        ("false_positive_rate_normal_anomaly_label", "Normal-event false-positive rate", "#E09F3E"),
    )
    positions = np.arange(len(names))
    offsets = np.linspace(-0.22, 0.22, len(metrics))
    fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
    for offset, (key, label, color) in zip(offsets, metrics, strict=True):
        values = [models[name][key] for name in names]
        bars = ax.barh(positions + offset, values, height=0.19, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    ax.set_yticks(positions, ["Default Isolation Forest", "Tuned Isolation Forest"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Score or rate")
    ax.set_title("Anomaly models · one-time locked synthetic evaluation")
    ax.grid(axis="x", color="#d9dee5", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.savefig(OUT / "readme_anomaly_model_metrics.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    risk_model_plot()
    confusion_matrix_plot()
    anomaly_model_plot()
